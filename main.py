from agent.service_agent import ServiceAgent
from tools.knowledge_tools import query_knowledge

def main():
  # 当前只是命令行演示，模拟用户已经完成登录。
  current_user_id = "USR001"
  customer_service_agent = ServiceAgent(current_user_id=current_user_id)
  print('---------------------------电商智能客服系统启动成功-------------------------------')
  print('您可以询问：')
  print('- 订单状态：我的订单OD1001状态如何')
  print('- 退货：订单OD1001退货')
  print('- 投诉：订单OD1001投诉商品质量')
  print('- 查询知识库：请问退货政策是什么？')
  print('- 输入：exit 退出系统')

  while True:
    user_input = input('请输入你的问题：')
    if user_input.lower() == 'exit':
      print('系统已退出')
      break
    if user_input:
      rs = customer_service_agent.chat(user_input)
      print(f'客服：{rs}')

'''
1. 修改提示词
2. 修改工具的描述
3. 更换LLM
4. 更换智能体的代理：create_react_agent，create_tool_calling_agent，create_openai_tools_agent
'''


if __name__ == '__main__':
  main()