#!/usr/bin/env python3
"""
测试流式响应功能
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_agent.config import Config
from rag_agent.main import RAGAgent
from rag_agent.streaming_handler import EventType


async def test_streaming():
    """
    测试流式响应功能
    """
    print("开始测试流式响应功能...")
    
    # 加载配置
    config = Config()
    
    # 创建RAG智能体
    agent = RAGAgent(config)
    
    # 测试流式查询
    user_question = "什么是人工智能？"
    print(f"\n测试问题: {user_question}")
    print("流式响应开始:\n")
    
    # 收集完整的响应
    full_response = ""
    
    try:
        # 异步遍历流式响应
        async for event in agent.query_stream(user_question):
            if event.event == EventType.LOADING:
                print(f"[LOADING] {event.content}")
            elif event.event == EventType.THINK:
                print(f"[THINK] {event.content}")
            elif event.event == EventType.TEXT:
                print(f"[TEXT] {event.content}", end="", flush=True)
                full_response += event.content
            elif event.event == EventType.REFERENCE_DOC:
                print(f"[REFERENCE_DOC] {event.content}")
                if event.documents:
                    print(f"  相关文档数量: {len(event.documents)}")
            elif event.event == EventType.ERROR:
                print(f"[ERROR] {event.content}")
            elif event.event == EventType.DONE:
                print(f"\n[DONE] {event.content}")
        
        print(f"\n\n完整响应: {full_response}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_streaming())