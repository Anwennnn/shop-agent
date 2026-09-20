from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_demo_renders_core_controls():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert len(app.title) == 1
    assert "电商智能客服 Agent" in app.title[0].value
    assert app.selectbox[0].value == "U1001"
    assert hasattr(app.session_state["agent"], "agent_executor")

    button_labels = {button.label for button in app.button}
    assert {
        "📦 查询订单",
        "↩️ 申请退货",
        "📝 提交投诉",
        "📚 商城政策",
        "🧹 清空对话",
        "↻ 重置演示数据",
    }.issubset(button_labels)

    page_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "当前用户：**U1001**" in page_text
    assert "订单查询" in page_text
    assert "退货申请" in page_text
    assert "投诉提交" in page_text
    assert "商城政策问答" in page_text


def test_switching_demo_user_rebuilds_the_session():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox[0].set_value("U1002").run()

    assert not app.exception
    assert app.selectbox[0].value == "U1002"
    page_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "当前用户：**U1002**" in page_text
    assert "示例订单：**OD1002**" in page_text


def test_pending_action_shows_confirmation_controls():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    next(button for button in app.button if button.label == "📝 提交投诉").click().run()

    assert not app.exception
    button_labels = {button.label for button in app.button}
    assert "✅ 确认执行" in button_labels
    assert "❌ 取消操作" in button_labels
    assert app.session_state["agent"].pending_action == {
        "type": "complain_order",
        "order_id": "OD1001",
        "complaint_content": "收到的商品包装破损",
    }
