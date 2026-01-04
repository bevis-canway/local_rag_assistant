#!/usr/bin/env python3
"""
使用多知识库示例
演示如何使用配置好的多个知识库进行查询
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_agent.config import Config
from rag_agent.main import RAGAgent
from rag_agent.streaming_handler import EventType


def setup_agent_with_multi_kbs():
    """
    设置支持多知识库的智能体
    """
    print("=== 初始化支持多知识库的小魔仙RAG智能体 ===\n")
    
    # 加载配置
    config = Config()
    
    # 创建RAG智能体（自动处理多知识库）
    agent = RAGAgent(config)
    
    # 检查当前配置的知识库
    kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
    print(f"当前配置了 {len(kb_infos)} 个知识库:")
    for kb_info in kb_infos:
        status = "启用" if kb_info.enabled else "禁用"
        print(f"  - {kb_info.name} ({status}): {kb_info.description}")
        print(f"    路径: {kb_info.path}")
        print(f"    文档数: {kb_info.indexed_documents_count}")
    
    return agent


def test_regular_query(agent, question):
    """
    测试常规查询（跨多知识库）
    """
    print(f"\n=== 常规查询测试 ===")
    print(f"问题: {question}")
    
    try:
        response = agent.query(question)
        print(f"回答: {response}")
    except Exception as e:
        print(f"查询出错: {e}")


async def test_streaming_query(agent, question):
    """
    测试流式查询（跨多知识库）
    """
    print(f"\n=== 流式查询测试 ===")
    print(f"问题: {question}")
    print("流式响应:")
    
    try:
        full_response = ""
        async for event in agent.query_stream(question):
            if event.event == EventType.LOADING:
                print(f"[LOADING] {event.content}")
            elif event.event == EventType.THINK:
                print(f"[THINK] {event.content}")
            elif event.event == EventType.TEXT:
                print(f"[TEXT] {event.content}", end="", flush=True)
                full_response += event.content
            elif event.event == EventType.REFERENCE_DOC:
                print(f"\n[REFERENCE_DOC] {event.content}")
                if event.documents:
                    print(f"  相关文档数量: {len(event.documents)}")
                    for i, doc in enumerate(event.documents[:3]):  # 只显示前3个文档信息
                        kb_name = doc.get('metadata', {}).get('knowledge_base', 'unknown')
                        title = doc.get('metadata', {}).get('title', 'Unknown')
                        print(f"    {i+1}. 知识库: {kb_name}, 标题: {title}")
                    if len(event.documents) > 3:
                        print(f"    ... 还有 {len(event.documents) - 3} 个文档")
            elif event.event == EventType.ERROR:
                print(f"[ERROR] {event.content}")
            elif event.event == EventType.DONE:
                print(f"\n[DONE] {event.content}")
        
        print(f"\n\n完整回答: {full_response}")
        
    except Exception as e:
        print(f"流式查询出错: {e}")


def show_knowledge_base_management(agent):
    """
    展示知识库管理功能
    """
    print(f"\n=== 知识库管理功能 ===")
    
    # 列出所有知识库
    print("1. 列出所有知识库:")
    kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
    for kb_info in kb_infos:
        status = "启用" if kb_info.enabled else "禁用"
        print(f"   - {kb_info.name} ({status})")
    
    # 添加新知识库示例（如果需要）
    print("\n2. 添加新知识库示例:")
    print("   # agent.knowledge_base_manager.add_knowledge_base(kb_config)")
    
    # 索引特定知识库示例
    print("\n3. 索引知识库示例:")
    print("   # agent.knowledge_base_manager.index_knowledge_base('kb_name')")


async def main():
    """
    主函数 - 演示多知识库使用
    """
    print("小魔仙RAG智能体 - 多知识库使用演示")
    print("=" * 50)
    
    # 设置智能体
    agent = setup_agent_with_multi_kbs()
    
    if not agent.knowledge_base_manager.knowledge_bases:
        print("\n警告: 没有配置任何知识库，请先运行配置脚本")
        return
    
    # 测试问题
    test_questions = [
        "请总结一下这些知识库中的主要内容",
        "这些知识库中有什么共同点和不同点",
    ]
    
    # 执行常规查询测试
    for question in test_questions:
        test_regular_query(agent, question)
    
    # 执行流式查询测试
    for question in test_questions:
        await test_streaming_query(agent, question)
    
    # 展示管理功能
    show_knowledge_base_management(agent)
    
    print(f"\n" + "=" * 50)
    print("多知识库使用演示完成！")


if __name__ == "__main__":
    asyncio.run(main())