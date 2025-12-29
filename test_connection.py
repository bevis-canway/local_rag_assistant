#!/usr/bin/env python3
"""
测试Ollama连接和API嵌入功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


import httpx

from rag_agent.config import Config
from rag_agent.main import RAGAgent


def test_ollama_connection():
    """测试Ollama服务连接"""
    print("测试Ollama服务连接...")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                print("✓ Ollama服务连接成功")
                print(
                    f"  可用模型: {[model['name'] for model in models.get('models', [])]}"
                )
                return True
            else:
                print(f"✗ Ollama服务连接失败，状态码: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Ollama服务连接失败: {e}")
        return False


def test_api_embeddings():
    """测试API嵌入功能"""
    print("\n测试API嵌入功能...")
    try:
        # 设置配置
        config = Config()
        agent = RAGAgent(config)

        print("✓ RAG Agent初始化成功")
        print(f"  使用API嵌入: {agent.vector_store.use_api_embeddings}")
        print(f"  嵌入模型: {agent.vector_store.embedding_model}")

        # 尝试进行一次简单的嵌入调用（不实际执行，只验证配置）
        if agent.vector_store.use_api_embeddings:
            print("✓ API嵌入配置正确")
            return True
        else:
            print("✗ 未使用API嵌入")
            return False
    except Exception as e:
        print(f"✗ API嵌入测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_obsidian_connection():
    """测试Obsidian连接"""
    print("\n测试Obsidian连接...")
    try:
        config = Config()
        agent = RAGAgent(config)

        notes = agent.obsidian_connector.list_notes()
        print("✓ 成功连接Obsidian知识库")
        print(f"  笔记数量: {len(notes)}")
        return True
    except Exception as e:
        print(f"✗ Obsidian连接失败: {e}")
        return False


def main():
    print("RAG智能体连接测试")
    print("=" * 50)

    # 测试各项连接
    ollama_ok = test_ollama_connection()
    embeddings_ok = test_api_embeddings()
    obsidian_ok = test_obsidian_connection()

    print("\n测试结果:")
    print(f"  Ollama连接: {'✓' if ollama_ok else '✗'}")
    print(f"  API嵌入: {'✓' if embeddings_ok else '✗'}")
    print(f"  Obsidian连接: {'✓' if obsidian_ok else '✗'}")

    if all([ollama_ok, embeddings_ok, obsidian_ok]):
        print("\n✓ 所有测试通过！RAG智能体配置正确。")
        print("  您现在可以运行 'python -m rag_agent.main' 启动智能体")
        return True
    else:
        print("\n✗ 部分测试失败，请检查配置")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
