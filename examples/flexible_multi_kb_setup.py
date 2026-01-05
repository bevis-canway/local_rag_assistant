#!/usr/bin/env python3
"""
灵活多知识库配置
用于配置多个知识库，支持从配置文件读取路径
"""
import os
import sys
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_agent.config import Config
from rag_agent.main import RAGAgent
from rag_agent.knowledge_base_manager import KnowledgeBaseConfig


def setup_multi_knowledge_bases(base_path=None):
    """
    配置多个知识库，支持从配置文件读取路径
    """
    print("=== 灵活配置多知识库 ===\n")
    
    # 如果没有指定路径，从配置中获取默认路径
    if base_path is None:
        config = Config()
        base_path = getattr(config, 'DEFAULT_KNOWLEDGE_BASE_PATH', '/Users/xiejindong/Desktop/multi_knowledge_base')
    
    if not os.path.exists(base_path):
        print(f"错误: 目录 {base_path} 不存在")
        print(f"请确保知识库路径正确，或在 .env 文件中设置 DEFAULT_KNOWLEDGE_BASE_PATH")
        return
    
    print(f"在 {base_path} 中查找知识库...")
    
    # 列出目录中的子目录
    subdirs = [d for d in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"找到 {len(subdirs)} 个子目录:")
    for i, subdir in enumerate(subdirs, 1):
        subdir_path = os.path.join(base_path, subdir)
        print(f"  {i}. {subdir} (路径: {subdir_path})")
    
    if not subdirs:
        print("未找到任何子目录作为知识库")
        return
    
    # 加载配置
    config = Config()
    
    # 创建RAG智能体（自动初始化知识库管理器）
    agent = RAGAgent(config)
    
    # 清除默认知识库（可选，如果只想要自动发现的知识库）
    # agent.knowledge_base_manager.knowledge_bases = {}
    
    # 为每个子目录创建知识库配置
    for i, subdir in enumerate(subdirs, 1):
        kb_name = f"kb_{subdir.lower().replace(' ', '_').replace('-', '_')}"  # 规范化名称
        subdir_path = os.path.join(base_path, subdir)
        
        kb_config = KnowledgeBaseConfig(
            name=kb_name,
            type="obsidian",  # 假设都是Obsidian知识库，根据实际情况调整
            path=subdir_path,
            description=f"{subdir} 知识库",
            enabled=True,
            vector_store_path=f"./vector_store/{kb_name}"  # 每个知识库独立的向量存储
        )
        
        # 添加知识库
        success = agent.knowledge_base_manager.add_knowledge_base(kb_config)
        if success:
            print(f"  ✓ 成功添加知识库: {kb_name}")
            print(f"    路径: {subdir_path}")
            print(f"    类型: {kb_config.type}")
            print(f"    描述: {kb_config.description}")
        else:
            print(f"  ✗ 添加知识库 {kb_name} 失败")
    
    print(f"\n知识库配置完成！当前共有 {len(agent.knowledge_base_manager.knowledge_bases)} 个知识库。")
    
    # 列出所有配置的知识库
    print("\n所有配置的知识库:")
    kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
    for kb_info in kb_infos:
        status = "启用" if kb_info.enabled else "禁用"
        print(f"  - {kb_info.name} ({status}): {kb_info.description}")
        print(f"    路径: {kb_info.path}")
        print(f"    文档数: {kb_info.indexed_documents_count}")
    
    # 保存配置以便后续使用（仅配置，不索引）
    config_file = os.path.join(project_root, "multi_kb_config.json")
    agent.knowledge_base_manager.save_configs(config_file)
    print(f"\n配置已保存到: {config_file}")
    
    print("\n=== 配置完成 ===")
    print("您可以使用以下命令来索引知识库:")
    print(f"python -c \"from examples.flexible_multi_kb_setup import index_knowledge_bases; index_knowledge_bases()\"")
    
    return agent


def index_knowledge_bases():
    """
    单独的索引函数，可以在服务稳定时运行
    """
    print("=== 开始索引知识库 ===\n")
    
    # 加载配置
    config = Config()
    agent = RAGAgent(config)
    
    # 从文件加载知识库配置
    config_file = os.path.join(
        Path(__file__).parent.parent,
        "multi_kb_config.json"
    )
    
    if os.path.exists(config_file):
        agent.knowledge_base_manager.load_configs(config_file)
        print("已从配置文件加载知识库设置")
    else:
        print("未找到配置文件，使用默认配置")
    
    # 索引所有知识库
    print("开始索引所有知识库...")
    index_results = agent.knowledge_base_manager.index_all_knowledge_bases()
    
    print("\n索引结果:")
    for kb_name, success in index_results.items():
        status = "成功" if success else "失败"
        print(f"  - {kb_name}: {status}")
    
    print("\n索引完成！")


def show_usage_instructions():
    """
    显示使用说明
    """
    print("\n=== 使用说明 ===\n")
    print("1. 配置知识库（使用默认路径）:")
    print("   python examples/flexible_multi_kb_setup.py")
    print()
    print("2. 配置知识库（指定路径）:")
    print("   python -c \"from examples.flexible_multi_kb_setup import setup_multi_knowledge_bases; setup_multi_knowledge_bases('/your/custom/path')\"")
    print()
    print("3. 索引知识库（确保Ollama服务正在运行）:")
    print("   python -c \"from examples.flexible_multi_kb_setup import index_knowledge_bases; index_knowledge_bases()\"")
    print()
    print("4. 通过环境变量配置（在 .env 文件中）:")
    print("   DEFAULT_KNOWLEDGE_BASE_PATH=/your/custom/path")
    print()
    print("5. 通过环境变量配置多个特定知识库:")
    kb_config_example = [
        {
            "name": "project_docs",
            "type": "obsidian",
            "path": "/path/to/project/docs",
            "description": "项目文档知识库",
            "enabled": True,
            "vector_store_path": "./vector_store/project_docs"
        },
        {
            "name": "research_docs",
            "type": "obsidian",
            "path": "/path/to/research/docs",
            "description": "研究文档知识库",
            "enabled": True,
            "vector_store_path": "./vector_store/research_docs"
        }
    ]
    print(f"   KNOWLEDGE_BASES_CONFIG='{json.dumps(kb_config_example, ensure_ascii=False, indent=2)}'")
    print()
    print("6. 在代码中使用多知识库:")
    print("   from rag_agent.main import RAGAgent")
    print("   agent = RAGAgent(config)")
    print("   # 查询会自动跨所有启用的知识库进行")


def scan_and_configure_knowledge_bases(base_path):
    """
    扫描并配置指定路径下的知识库
    """
    print(f"=== 扫描并配置 {base_path} 下的知识库 ===\n")
    
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
        return
    
    # 检查是否是有效的知识库目录（包含笔记文件）
    def is_valid_knowledge_base(path):
        # 检查目录是否包含markdown文件或其他有效文档
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.md', '.txt', '.pdf', '.docx')):
                    return True
        return False
    
    if is_valid_knowledge_base(base_path):
        print(f"发现有效知识库: {base_path}")
        config = Config()
        agent = RAGAgent(config)
        
        # 使用目录名作为知识库名
        kb_name = os.path.basename(os.path.normpath(base_path)).lower().replace(' ', '_').replace('-', '_')
        
        kb_config = KnowledgeBaseConfig(
            name=f"kb_{kb_name}",
            type="obsidian",  # 可根据实际类型调整
            path=base_path,
            description=f"{os.path.basename(base_path)} 知识库",
            enabled=True,
            vector_store_path=f"./vector_store/kb_{kb_name}"
        )
        
        success = agent.knowledge_base_manager.add_knowledge_base(kb_config)
        if success:
            print(f"  ✓ 成功添加知识库: {kb_config.name}")
            print(f"    路径: {kb_config.path}")
            print(f"    类型: {kb_config.type}")
            print(f"    描述: {kb_config.description}")
            
            # 保存配置
            config_file = os.path.join(Path(__file__).parent.parent, "multi_kb_config.json")
            agent.knowledge_base_manager.save_configs(config_file)
            print(f"    配置已保存到: {config_file}")
        else:
            print(f"  ✗ 添加知识库 {kb_config.name} 失败")
        
        return agent
    else:
        print(f"路径 {base_path} 不包含有效的知识库文件")
        return None


if __name__ == "__main__":
    print("小魔仙RAG智能体 - 灵活多知识库配置")
    print("=" * 50)
    
    # 执行灵活配置
    agent = setup_multi_knowledge_bases()
    
    # 显示使用说明
    show_usage_instructions()
    
    print("\n" + "=" * 50)
    print("多知识库配置完成！")
    print("注意：为避免长时间索引，此脚本跳过了索引步骤。")
    print("您可以在Ollama服务稳定运行后单独执行索引。")