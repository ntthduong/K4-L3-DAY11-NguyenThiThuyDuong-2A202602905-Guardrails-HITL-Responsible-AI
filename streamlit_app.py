"""Local Streamlit demo for the guarded VinBank Blue agent."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.agent import create_blue_agent  # noqa: E402
from assignment.pipeline import build_production_plugins  # noqa: E402
from core.config import blue_provider_label, get_openrouter_api_key  # noqa: E402
from core.utils import chat_with_agent  # noqa: E402


MAX_PROMPT_LENGTH = 4_000


@st.cache_resource
def build_demo_runtime():
    """Create one local runtime and preserve rate-limit state across reruns."""
    plugins = build_production_plugins(
        max_requests=10,
        window_seconds=60,
        use_llm_judge=False,
    )
    agent, runner = create_blue_agent(plugins=plugins)
    return agent, runner, plugins


def plugin_by_name(plugins: list, name: str):
    return next(plugin for plugin in plugins if getattr(plugin, "name", "") == name)


async def chat_with_retry(agent, runner, prompt: str) -> str:
    """Retry one transient OpenRouter rate limit for the local demo."""
    from openai import RateLimitError

    for attempt in range(2):
        try:
            response, _ = await chat_with_agent(agent, runner, prompt)
            return response
        except RateLimitError as exc:
            if attempt == 1:
                raise
            headers = getattr(getattr(exc, "response", None), "headers", None)
            retry_after = headers.get("retry-after") if headers is not None else None
            try:
                wait_seconds = float(retry_after) if retry_after else 60.0
            except (TypeError, ValueError):
                wait_seconds = 60.0
            wait_seconds = max(5.0, min(wait_seconds, 90.0))
            status = st.empty()
            status.info(
                f"OpenRouter đang giới hạn tốc độ. Thử lại sau {wait_seconds:.0f} giây…"
            )
            await asyncio.sleep(wait_seconds)
            status.empty()
    raise RuntimeError("unreachable")


def run_prompt(agent, runner, plugins: list, prompt: str) -> dict:
    """Run one prompt and report which deterministic layer acted on it."""
    rate = plugin_by_name(plugins, "rate_limiter")
    input_guard = plugin_by_name(plugins, "input_guardrail")
    output_guard = plugin_by_name(plugins, "output_guardrail")
    before = {
        "rate": rate.blocked_count,
        "input": input_guard.blocked_count,
        "redacted": output_guard.redacted_count,
        "output": output_guard.blocked_count,
    }

    response = asyncio.run(chat_with_retry(agent, runner, prompt))
    layer = None
    blocked = False
    redacted = False
    if rate.blocked_count > before["rate"]:
        layer, blocked = "rate_limiter", True
    elif input_guard.blocked_count > before["input"]:
        layer, blocked = "input_guardrail", True
    elif (
        output_guard.redacted_count > before["redacted"]
        or output_guard.blocked_count > before["output"]
    ):
        layer, blocked, redacted = "output_guardrail", True, True

    return {
        "response": response,
        "blocked": blocked,
        "redacted": redacted,
        "layer": layer,
    }


st.set_page_config(
    page_title="VinBank Guardrails Demo",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ VinBank Guardrails Demo")
st.caption(f"Blue Agent · `{blue_provider_label()}` · local lab demo")
st.warning(
    "Chỉ dùng dữ liệu giả cho buổi lab. Không nhập API key, mật khẩu hoặc thông tin "
    "ngân hàng thật. Câu trả lời của mô hình có thể không chính xác."
)

if not get_openrouter_api_key():
    st.error("Thiếu OPENROUTER_API_KEY trong file .env.")
    st.stop()

agent, runner, plugins = build_demo_runtime()
rate_plugin = plugin_by_name(plugins, "rate_limiter")
input_plugin = plugin_by_name(plugins, "input_guardrail")
output_plugin = plugin_by_name(plugins, "output_guardrail")

with st.sidebar:
    st.header("Demo controls")
    st.code(blue_provider_label(), language=None)
    st.metric("Requests", rate_plugin.total_count)
    st.metric("Input blocks", input_plugin.blocked_count)
    st.metric("Output redactions", output_plugin.redacted_count)
    if st.button("Reset demo", use_container_width=True):
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Nhập câu hỏi banking hoặc prompt kiểm thử…")
if prompt:
    prompt = prompt.strip()
    if not prompt:
        st.error("Prompt không được để trống.")
        st.stop()
    if len(prompt) > MAX_PROMPT_LENGTH:
        st.error(f"Prompt quá dài. Giới hạn là {MAX_PROMPT_LENGTH:,} ký tự.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Blue Agent đang xử lý…"):
                result = run_prompt(agent, runner, plugins, prompt)
            st.markdown(result["response"])
            if result["blocked"]:
                meta = f"BLOCK · {result['layer']}"
                if result["redacted"]:
                    meta += " · sensitive output redacted"
            else:
                meta = "ALLOW · passed guardrails"
            st.caption(meta)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "meta": meta,
            })
        except Exception as exc:
            error_name = type(exc).__name__
            st.error(
                "Không thể gọi Blue model lúc này. Hãy kiểm tra OpenRouter hoặc thử lại sau. "
                f"({error_name})"
            )
