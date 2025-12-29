"""
测试脚本，用于验证RAG智能体Demo的基本功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_agent.config import Config
from rag_agent.obsidian_connector import ObsidianConnector
from rag_agent.prompt_engineer import PromptEngineer
from rag_agent.retriever import Retriever
from rag_agent.vector_store import VectorStore


def test_config():
    """测试配置加载"""
    print("测试配置加载...")
    config = Config()
    print(f"配置加载成功: {config.OLLAMA_MODEL}")
    return True


def test_obsidian_connector():
    """测试Obsidian连接器（如果配置了路径）"""
    print("测试Obsidian连接器...")
    config = Config()
    if config.OBSIDIAN_VAULT_PATH:
        connector = ObsidianConnector(
            vault_path=config.OBSIDIAN_VAULT_PATH,
            api_url=config.OBSIDIAN_API_URL,
            api_key=config.OBSIDIAN_API_KEY,
        )
        notes = connector.list_notes()
        print(f"找到 {len(notes)} 个笔记")
        if notes:
            content = connector.get_note_content(notes[0]["id"])
            print(f"第一个笔记内容长度: {len(content)}")
        return True
    else:
        print("未配置OBSIDIAN_VAULT_PATH，跳过测试")
        return True


def test_vector_store():
    """测试向量存储"""
    print("测试向量存储...")
    vector_store = VectorStore()

    # 添加测试文档
    test_docs = [
        {
            "id": "test1",
            "content": "这是一个关于人工智能的测试文档。",
            "metadata": {"title": "AI文档", "path": "/test/ai.md"},
        },
        {
            "id": "test2",
            "content": "这是一个关于机器学习的测试文档。",
            "metadata": {"title": "ML文档", "path": "/test/ml.md"},
        },
    ]

    vector_store.add_documents(test_docs)
    print("文档添加成功")

    # 搜索测试
    results = vector_store.search("人工智能", top_k=1)
    print(f"搜索结果数量: {len(results)}")
    if results:
        print(f"匹配内容: {results[0]['content']}")
        print(f"相似度: {results[0]['similarity']}")

    return True


def test_retriever():
    """测试检索器"""
    print("测试检索器...")
    vector_store = VectorStore()
    retriever = Retriever(vector_store, top_k=2)

    # 添加测试数据
    test_docs = [
        {
            "id": "test1",
            "content": "Python是一种高级编程语言，广泛用于数据科学和人工智能领域。",
            "metadata": {"title": "Python介绍", "path": "/test/python.md"},
        },
        {
            "id": "test2",
            "content": "机器学习是人工智能的一个分支，涉及算法和统计模型。",
            "metadata": {"title": "机器学习", "path": "/test/ml.md"},
        },
    ]

    vector_store.add_documents(test_docs)

    # 检索测试
    results = retriever.retrieve("Python编程语言")
    print(f"检索到 {len(results)} 个结果")

    formatted = retriever.retrieve_and_format("Python编程语言")
    print(f"格式化结果:\n{formatted}")

    return True


def test_prompt_engineer():
    """测试提示工程"""
    print("测试提示工程...")
    pe = PromptEngineer()

    context = "Python是一种高级编程语言，广泛用于数据科学和人工智能领域。"
    query = "什么是Python？"

    prompt = pe.build_rag_prompt(query, context)
    print(f"生成的提示词长度: {len(prompt)}")
    print(f"提示词预览: {prompt[:100]}...")

    return True


def main():
    """运行所有测试"""
    print("开始测试RAG智能体Demo...")

    tests = [
        ("配置加载", test_config),
        ("向量存储", test_vector_store),
        ("检索器", test_retriever),
        ("提示工程", test_prompt_engineer),
        ("Obsidian连接器", test_obsidian_connector),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n--- 运行测试: {name} ---")
            if test_func():
                print(f"✓ {name} 测试通过")
                passed += 1
            else:
                print(f"✗ {name} 测试失败")
        except Exception as e:
            print(f"✗ {name} 测试出错: {e}")

    print("\n--- 测试总结 ---")
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("所有测试通过！")
        return True
    else:
        print(f"有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    main()
