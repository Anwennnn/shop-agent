"""Command-line entry point for the customer-service agent."""

from __future__ import annotations

import time

from agent.service_agent import ServiceAgent
from utils.observability import (
    agent_error_message,
    classify_error,
    configure_logging,
    log_event,
    new_request_id,
    reset_request_id,
    set_request_id,
)


def main() -> None:
    configure_logging()
    current_user_id = "USR001"  # CLI demo: simulate an authenticated session.

    startup_request_id = new_request_id()
    token = set_request_id(startup_request_id)
    started_at = time.perf_counter()
    try:
        customer_service_agent = ServiceAgent(current_user_id=current_user_id)
        log_event(
            component="application",
            operation="startup",
            success=True,
            started_at=started_at,
        )
    except Exception as exc:
        error_type = classify_error(exc)
        log_event(
            component="application",
            operation="startup",
            success=False,
            started_at=started_at,
            error_type=error_type,
        )
        print(agent_error_message(error_type, startup_request_id))
        return
    finally:
        reset_request_id(token)

    print("------------------电商智能客服系统启动成功------------------")
    print("您可以询问：")
    print("- 订单状态：我的订单 OD1001 状态如何")
    print("- 退货：订单 OD1003 退货，原因是商品质量问题")
    print("- 投诉：投诉订单 OD1001 商品质量问题")
    print("- 知识库：请问退货政策是什么？")
    print("- 对退货和投诉操作，请根据提示回复“确认”或“取消”")
    print("- 输入 exit 退出系统")

    while True:
        try:
            user_input = input("请输入您的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n系统已退出")
            break

        if user_input.lower() == "exit":
            print("系统已退出")
            break

        response = customer_service_agent.chat(user_input)
        print(f"客服：{response}")


if __name__ == "__main__":
    main()
