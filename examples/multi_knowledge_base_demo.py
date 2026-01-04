#!/usr/bin/env python3
"""
多知识库功能演示
展示小魔仙RAG智能体的多知识库支持功能
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_agent.config import Config
from rag_agent.main import RAGAgent
from rag_agent.knowledge_base_manager import KnowledgeBaseConfig


def demo_basic_multi_kb():
    """
    演示基本的多知识库功能
    """
    print("=== 小魔仙RAG智能体 - 多知识库功能演示 ===\n")
    
    # 加载配置
    config = Config()
    
    # 创建RAG智能体（自动初始化知识库管理器）
    agent = RAGAgent(config)
    
    print("1. 初始知识库列表:")
    kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
    for kb_info in kb_infos:
        print(f"   - {kb_info.name}: {kb_info.description} (文档数: {kb_info.indexed_documents_count})")
    
    print("\n2. 添加新的知识库:")
    # 添加一个示例知识库
    sample_kb = KnowledgeBaseConfig(
        name="sample_project",
        type="obsidian",
        path="/path/to/sample/project",  # 实际使用时替换为真实路径
        description="示例项目知识库",
        enabled=True
    )
    
    # 注意：在实际使用中，你需要确保路径存在
    # 这里只是演示API的使用
    print(f"   尝试添加知识库: {sample_kb.name}")
    print(f"   类型: {sample_kb.type}")
    print(f"   路径: {sample_kb.path}")
    print(f"   描述: {sample_kb.description}")
    
    print("\n3. 多知识库查询演示:")
    print("   在实际使用中，查询会自动跨所有启用的知识库进行检索")
    print("   系统会整合来自不同知识库的相关文档以生成最佳回答")
    
    print("\n4. 知识库管理功能:")
    print("   - 支持动态添加/移除知识库")
    print("   - 支持启用/禁用特定知识库")
    print("   - 支持配置不同知识库的存储路径")
    print("   - 支持跨知识库检索和索引")
    
    print("\n=== 演示完成 ===")


async def demo_streaming_with_multi_kb():
    """
    演示流式响应与多知识库的结合
    """
    print("\n=== 流式响应与多知识库结合演示 ===\n")
    
    config = Config()
    agent = RAGAgent(config)
    
    # 演示如何使用流式查询（实际使用时需要有效的查询）
    print("流式查询将自动利用多知识库功能:")
    print("- THINK事件: 显示正在检索多个知识库")
    print("- REFERENCE_DOC事件: 显示来自不同知识库的文档")
    print("- TEXT事件: 显示生成的回答内容")
    
    print("\n实际使用示例:")
    print("async for event in agent.query_stream('您的问题'):")
    print("    if event.event == EventType.REFERENCE_DOC:")
    print("        print(f'找到 {len(event.documents)} 个相关文档')")
    print("        # 文档可能来自不同知识库")


def demo_configuration():
    """
    演示多知识库配置
    """
    print("\n=== 多知识库配置演示 ===\n")
    
    print("1. 环境变量配置:")
    print("   KNOWLEDGE_BASES_CONFIG='[{\"name\": \"project_a\", \"type\": \"obsidian\", \"path\": \"/path/to/project_a\", \"description\": \"项目A知识库\", \"enabled\": true}]'")
    
    print("\n2. 代码中动态配置:")
    print("   kb_config = KnowledgeBaseConfig(")
    print("       name=\"tech_docs\",")
    print("       type=\"obsidian\",")
    print("       path=\"/path/to/tech/docs\",")
    print("       description=\"技术文档知识库\",")
    print("       enabled=True")
    print("   )")
    print("   agent.knowledge_base_manager.add_knowledge_base(kb_config)")
    
    print("\n3. 配置持久化:")
    print("   # 保存配置到文件")
    print("   agent.knowledge_base_manager.save_configs('kb_configs.json')")
    print("   # 从文件加载配置")
    print("   agent.knowledge_base_manager.load_configs('kb_configs.json')")


if __name__ == "__main__":
    print("小魔仙RAG智能体 - 多知识库功能演示")
    print("=" * 50)
    
    # 运行基本演示
    demo_basic_multi_kb()
    
    # 运行配置演示
    demo_configuration()
    
    # 运行流式响应演示
    asyncio.run(demo_streaming_with_multi_kb())
    
    print("\n" + "=" * 50)
    print("多知识库功能已成功集成到小魔仙RAG智能体中！")