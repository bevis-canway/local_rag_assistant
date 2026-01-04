"""
小魔仙RAG智能体 - 幻觉缓解功能使用示例

本示例展示如何使用新添加的幻觉缓解功能
"""
from rag_agent.main import RAGAgent
from rag_agent.config import Config
from rag_agent.hallucination_detector import HallucinationDetector


def example_basic_usage():
    """基本使用示例"""
    print("=== 小魔仙RAG智能体 - 幻觉缓解功能演示 ===\n")
    
    # 初始化配置和智能体
    config = Config()
    agent = RAGAgent(config)
    
    print("智能体初始化完成，幻觉检测器已集成")
    print(f"生成参数: {config.get_generation_options()}")
    print()


def example_hallucination_detection():
    """幻觉检测功能示例"""
    print("=== 幻觉检测功能示例 ===\n")
    
    config = Config()
    detector = HallucinationDetector(config)
    
    # 测试一致的回答
    print("1. 测试一致的回答:")
    response = "According to the document, Python is a programming language."
    retrieved_docs = [
        {
            "content": "Python is a high-level programming language, originally created by Guido van Rossum in 1989.",
            "metadata": {"title": "Python Introduction", "path": "/python/intro.md"}
        }
    ]
    query = "What is Python?"
    
    result = detector.detect_hallucinations(response, retrieved_docs, query)
    print(f"   回答: {response}")
    print(f"   一致性: {result.is_consistent}")
    print(f"   置信度: {result.confidence_score}")
    print(f"   不一致项: {result.inconsistencies}")
    print()
    
    # 测试不一致的回答
    print("2. 测试不一致的回答:")
    response = "According to the document, Java was invented in 1990 by Bill Gates as a programming language."
    retrieved_docs = [
        {
            "content": "Java is an object-oriented programming language developed by James Gosling at Sun Microsystems in 1995.",
            "metadata": {"title": "Java Introduction", "path": "/java/intro.md"}
        }
    ]
    query = "What is Java?"
    
    result = detector.detect_hallucinations(response, retrieved_docs, query)
    print(f"   回答: {response}")
    print(f"   一致性: {result.is_consistent}")
    print(f"   置信度: {result.confidence_score}")
    print(f"   不一致项: {result.inconsistencies}")
    print()


def example_configured_generation():
    """配置生成参数示例"""
    print("=== 配置生成参数示例 ===\n")
    
    config = Config()
    print(f"温度 (Temperature): {config.GENERATION_TEMPERATURE} (较低值减少随机性)")
    print(f"Top-P: {config.GENERATION_TOP_P}")
    print(f"Top-K: {config.GENERATION_TOP_K}")
    print("这些参数配置有助于减少模型生成时的幻觉现象")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_hallucination_detection()
    example_configured_generation()
    
    print("=== 总结 ===")
    print("小魔仙RAG智能体现在具备以下幻觉缓解功能：")
    print("1. 实时幻觉检测 - 自动检测回答与文档的一致性")
    print("2. 生成参数优化 - 使用较低温度减少随机性")
    print("3. 防幻觉提示词 - 强制模型基于文档内容回答")
    print("4. 置信度评估 - 对回答准确性进行量化评估")