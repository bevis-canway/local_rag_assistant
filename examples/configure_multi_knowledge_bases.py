#!/usr/bin/env python3
"""
多知识库配置示例
演示如何配置位于 /Users/xiejindong/Desktop/multi_knowledge_base 的多个知识库
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


def setup_multi_knowledge_bases():
    """
    配置多个知识库
    """
    print("=== 配置多知识库示例 ===\n")
    
    # 检查目标目录是否存在
    base_path = "/Users/xiejindong/Desktop/multi_knowledge_base"
    if not os.path.exists(base_path):
        print(f"错误: 目录 {base_path} 不存在")
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
    
    # 清除默认知识库（可选，如果只需要自定义的知识库）
    # agent.knowledge_base_manager.knowledge_bases = {}
    
    print(f"\n开始配置 {len(subdirs)} 个知识库...")
    
    # 为每个子目录创建知识库配置
    for i, subdir in enumerate(subdirs, 1):
        kb_name = subdir.lower().replace(' ', '_').replace('-', '_')  # 规范化名称
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
    
    # 索引所有知识库
    print(f"\n开始索引所有知识库...")
    index_results = agent.knowledge_base_manager.index_all_knowledge_bases()
    
    print("\n索引结果:")
    for kb_name, success in index_results.items():
        status = "成功" if success else "失败"
        print(f"  - {kb_name}: {status}")
    
    # 保存配置以便后续使用
    config_file = os.path.join(project_root, "multi_kb_config.json")
    agent.knowledge_base_manager.save_configs(config_file)
    print(f"\n配置已保存到: {config_file}")
    
    print("\n=== 配置完成 ===")
    print("现在您的小魔仙RAG智能体可以访问多个知识库了！")
    print("查询时系统会自动跨所有启用的知识库进行检索。")


def load_existing_config():
    """
    从文件加载已保存的知识库配置
    """
    print("\n=== 从配置文件加载知识库 ===\n")
    
    config_file = os.path.join(
        Path(__file__).parent.parent,
        "multi_kb_config.json"
    )
    
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return None
    
    # 加载配置
    config = Config()
    agent = RAGAgent(config)
    
    # 从文件加载知识库配置
    agent.knowledge_base_manager.load_configs(config_file)
    
    print("成功从配置文件加载知识库配置:")
    kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
    for kb_info in kb_infos:
        status = "启用" if kb_info.enabled else "禁用"
        print(f"  - {kb_info.name} ({status}): {kb_info.description}")
    
    return agent


def query_with_multi_kbs():
    """
    演示跨多个知识库的查询
    """
    print("\n=== 跨多知识库查询演示 ===\n")
    
    # 尝试加载现有配置
    agent = load_existing_config()
    if not agent:
        print("未找到配置文件，先运行配置函数")
        return
    
    print("多知识库已就绪，查询将自动跨所有启用的知识库进行检索")
    print("例如，当您提问时，系统会：")
    print("1. 在所有启用的知识库中搜索相关文档")
    print("2. 整合来自不同知识库的相关信息")
    print("3. 生成综合性的回答")
    print("4. 在流式响应中显示来自不同知识库的参考文档")


def show_environment_config():
    """
    显示环境变量配置方式
    """
    print("\n=== 环境变量配置方式 ===\n")
    
    print("您也可以通过环境变量配置多知识库:")
    print("# 在 .env 文件中添加:")
    print("KNOWLEDGE_BASES_CONFIG='[")
    print("  {")
    print('    "name": "knowledge_base_1",')
    print('    "type": "obsidian",')
    print('    "path": "/Users/xiejindong/Desktop/multi_knowledge_base/knowledge_base_1",')
    print('    "description": "第一个知识库",')
    print('    "enabled": true,')
    print('    "vector_store_path": "./vector_store/kb1"')
    print("  },")
    print("  {")
    print('    "name": "knowledge_base_2",')
    print('    "type": "obsidian",')
    print('    "path": "/Users/xiejindong/Desktop/multi_knowledge_base/knowledge_base_2",')
    print('    "description": "第二个知识库",')
    print('    "enabled": true,')
    print('    "vector_store_path": "./vector_store/kb2"')
    print("  }")
    print("]'")
    
    print("\n或者直接设置环境变量:")
    kb_config = [
        {
            "name": "kb1",
            "type": "obsidian",
            "path": "/Users/xiejindong/Desktop/multi_knowledge_base/knowledge_base_1",
            "description": "第一个知识库",
            "enabled": True,
            "vector_store_path": "./vector_store/kb1"
        },
        {
            "name": "kb2",
            "type": "obsidian",
            "path": "/Users/xiejindong/Desktop/multi_knowledge_base/knowledge_base_2",
            "description": "第二个知识库",
            "enabled": True,
            "vector_store_path": "./vector_store/kb2"
        }
    ]
    
    print(f"export KNOWLEDGE_BASES_CONFIG='{json.dumps(kb_config, ensure_ascii=False, indent=2)}'")


if __name__ == "__main__":
    print("小魔仙RAG智能体 - 多知识库配置工具")
    print("=" * 50)
    
    # 配置多知识库
    setup_multi_knowledge_bases()
    
    # 显示环境变量配置方式
    show_environment_config()
    
    # 演示查询功能
    query_with_multi_kbs()
    
    print("\n" + "=" * 50)
    print("多知识库配置完成！")
    print("您现在可以使用小魔仙RAG智能体查询多个知识库了。")