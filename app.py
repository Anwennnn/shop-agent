"""Streamlit demo UI for the ecommerce customer-service Agent."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st

from agent.service_agent import ServiceAgent
from config import setting
from data.setup_database import setup_database


st.set_page_config(
    page_title="电商智能客服 Agent",
    page_icon="🛍️",
    layout="centered",
)


USER_OPTIONS = {
    "U1001": {
        "internal_id": "USR001",
        "orders": ["OD1001", "OD1003"],
        "return_order": "OD1003",
    },
    "U1002": {
        "internal_id": "USR002",
        "orders": ["OD1002"],
        "return_order": "OD1002",
    },
    "U1003": {
        "internal_id": "USR003",
        "orders": ["OD1004"],
        "return_order": "OD1004",
    },
}

WELCOME_MESSAGE = (
    "您好，我是电商智能客服。您可以查询订单、申请退货、提交投诉，"
    "也可以咨询商城政策。退货和投诉在您确认前不会执行。"
)


def _new_history() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def _demo_database_path() -> Path:
    if "demo_database_path" not in st.session_state:
        demo_directory = setting.DATA_DIR.parent / ".cache" / "streamlit_sessions"
        demo_directory.mkdir(parents=True, exist_ok=True)
        st.session_state.demo_database_path = (
            demo_directory / f"demo-{uuid4().hex}.db"
        )
    return Path(st.session_state.demo_database_path)


def _initialize_demo_database(*, reset: bool = False) -> Path:
    database_path = _demo_database_path()
    if reset and database_path.exists():
        database_path.unlink()
    if not database_path.exists():
        setup_database(
            json_path=setting.JSON_PATH,
            database_path=database_path,
        )
    return database_path


def _build_agent(display_user_id: str) -> ServiceAgent:
    user = USER_OPTIONS[display_user_id]
    return ServiceAgent(
        current_user_id=user["internal_id"],
        database_path=_initialize_demo_database(),
    )


def _reset_conversation() -> None:
    st.session_state.messages = _new_history()
    st.session_state.agent = _build_agent(st.session_state.selected_user)
    st.session_state.agent_user = st.session_state.selected_user
    st.session_state.last_error = None


def _reset_demo_data() -> None:
    _initialize_demo_database(reset=True)
    _reset_conversation()


def _ensure_session_state() -> None:
    st.session_state.setdefault("selected_user", "U1001")
    st.session_state.setdefault("messages", _new_history())
    st.session_state.setdefault("last_error", None)

    selected_user = st.session_state.selected_user
    if (
        "agent" not in st.session_state
        or st.session_state.get("agent_user") != selected_user
        or not hasattr(st.session_state.agent, "agent_executor")
        or not hasattr(st.session_state.agent, "tool_map")
    ):
        st.session_state.agent = _build_agent(selected_user)
        st.session_state.agent_user = selected_user
        st.session_state.messages = _new_history()


def _submit_message(message: str) -> None:
    clean_message = message.strip()
    if not clean_message:
        st.session_state.last_error = "请输入问题后再发送。"
        return

    st.session_state.messages.append({"role": "user", "content": clean_message})
    st.session_state.last_error = None
    try:
        with st.spinner("客服正在处理，请稍候……"):
            reply = st.session_state.agent.chat(clean_message)
    except Exception:
        reply = "系统暂时无法处理该请求，请稍后重试。"
        st.session_state.last_error = "请求处理失败，请稍后重试。"
    st.session_state.messages.append({"role": "assistant", "content": reply})


def _pending_action_text(action: dict) -> str:
    order_id = action.get("order_id", "未知订单")
    if action.get("type") == "return_order":
        return f"待确认：为订单 {order_id} 申请退货。"
    if action.get("type") == "complain_order":
        return f"待确认：针对订单 {order_id} 提交投诉。"
    return "当前有一项操作等待确认。"


try:
    _ensure_session_state()
except Exception:
    st.title("🛍️ 电商智能客服 Agent")
    st.error("系统初始化失败，请检查模型和环境配置后重试。")
    st.stop()


with st.sidebar:
    st.header("演示控制台")
    selected_user = st.selectbox(
        "模拟登录用户",
        options=list(USER_OPTIONS),
        key="selected_user",
        on_change=_reset_conversation,
    )
    selected = USER_OPTIONS[selected_user]
    st.markdown(f"当前用户：**{selected_user}**")
    st.markdown(f"示例订单：**{', '.join(selected['orders'])}**")

    st.divider()
    st.markdown("**系统能力**")
    st.markdown("✓ 订单查询")
    st.markdown("✓ 退货申请")
    st.markdown("✓ 投诉提交")
    st.markdown("✓ 商城政策问答")

    st.divider()
    if st.button("🧹 清空对话", use_container_width=True):
        _reset_conversation()
        st.rerun()
    if st.button("↻ 重置演示数据", use_container_width=True):
        _reset_demo_data()
        st.success("演示订单已经恢复。")
        st.rerun()


st.title("🛍️ 电商智能客服 Agent")
st.caption("订单查询 · 安全退货 · 投诉处理 · 商城政策问答")

st.subheader("示例问题")
query_order = selected["orders"][0]
return_order = selected["return_order"]
examples = [
    ("📦 查询订单", f"查询订单 {query_order}"),
    ("↩️ 申请退货", f"帮我退掉订单 {return_order}，原因是商品质量问题"),
    ("📝 提交投诉", f"投诉订单 {query_order}，收到的商品包装破损"),
    ("📚 商城政策", "七天无理由退货需要满足什么条件？"),
]

example_columns = st.columns(2)
example_prompt = None
for index, (label, prompt) in enumerate(examples):
    if example_columns[index % 2].button(label, key=f"example_{index}", use_container_width=True):
        example_prompt = prompt

if st.session_state.last_error:
    st.error(st.session_state.last_error)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending_action = st.session_state.agent.pending_action
if pending_action:
    st.warning(_pending_action_text(pending_action))
    confirm_column, cancel_column = st.columns(2)
    if confirm_column.button("✅ 确认执行", type="primary", use_container_width=True):
        _submit_message("确认")
        st.rerun()
    if cancel_column.button("❌ 取消操作", use_container_width=True):
        _submit_message("取消")
        st.rerun()

typed_prompt = st.chat_input("请输入问题，例如：查询订单 OD1001")
prompt_to_send = typed_prompt or example_prompt
if prompt_to_send:
    _submit_message(prompt_to_send)
    st.rerun()
