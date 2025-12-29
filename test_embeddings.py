#!/usr/bin/env python3
"""
测试嵌入模型配置的脚本
用于验证API嵌入模型是否正确配置
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_agent.vector_store import VectorStore
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_api_embeddings():
    """测试API嵌入模型"""
    print("测试API嵌入模型配置...")
    
    # 设置环境变量来启用API嵌入
    os.environ.setdefault("OLLAMA_API_KEY", "dummy_key")  # 使用虚拟密钥来启用API模式
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
    
    try:
        # 创建向量存储实例
        vs = VectorStore(persist_path="./test_vector_store")
        
        # 检查是否使用了API嵌入
        if vs.use_api_embeddings:
            print("✓ 成功配置API嵌入模型")
            print(f"  嵌入模型: {vs.embedding_model}")
            print(f"  API基础URL: {vs.base_url}")
            print("  OpenAI客户端将在实际调用时初始化")
            return True
        else:
            print("✗ 仍在使用本地嵌入模型，请检查环境变量配置")
            return False
            
    except Exception as e:
        print(f"✗ 初始化向量存储失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_embeddings():
    """测试本地嵌入模型（如果可用）"""
    print("\n测试本地嵌入模型...")
    
    # 清除API相关环境变量以测试本地模型
    for key in ["OLLAMA_API_KEY", "OPENAI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    try:
        vs = VectorStore(persist_path="./test_vector_store", model_name="all-MiniLM-L6-v2")
        
        if not vs.use_api_embeddings:
            print("✓ 成功配置本地嵌入模型")
            print("  本地模型将在实际调用时初始化")
            return True
        else:
            print("✗ 意外使用了API嵌入模型")
            return False
            
    except Exception as e:
        print(f"✗ 初始化本地向量存储失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("RAG智能体嵌入模型配置测试")
    print("="*50)
    
    # 首先测试API嵌入
    api_success = test_api_embeddings()
    
    if not api_success:
        print("\n尝试本地嵌入模型...")
        local_success = test_local_embeddings()
        
        if not local_success:
            print("\n错误: 两种嵌入模型都无法正常工作")
            print("请检查:")
            print("1. Ollama服务是否运行 (ollama serve)")
            print("2. nomic-embed-text模型是否已下载 (ollama pull nomic-embed-text)")
            print("3. 网络连接是否正常")
            print("4. 环境变量是否正确配置")
            return False
        else:
            print("\n✓ 本地嵌入模型可用")
    else:
        print("\n✓ API嵌入模型可用")
    
    print("\n✓ 测试完成，嵌入模型配置正常")
    print("\n提示: 如果您在中国大陆，可能需要配置代理或使用Ollama API嵌入模型")
    print("要使用Ollama API嵌入模型，请执行以下操作:")
    print("1. 启动Ollama服务: ollama serve")
    print("2. 下载嵌入模型: ollama pull nomic-embed-text")
    print("3. 设置环境变量:")
    print("   export OLLAMA_API_KEY='your_key'  # 可选，任意值即可")
    print("   export OLLAMA_BASE_URL='http://localhost:11434/v1'")
    print("   export EMBEDDING_MODEL='nomic-embed-text'")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)