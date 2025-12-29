import logging
import os
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ollama
import tiktoken

# 使用绝对导入替代相对导入
from rag_agent.config import Config
from rag_agent.obsidian_connector import ObsidianConnector
from rag_agent.prompt_engineer import PromptEngineer
from rag_agent.retriever import Retriever
from rag_agent.vector_store import VectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG智能体主类
    整合所有模块，提供问答功能
    参考aidev项目中的Agent实现
    """

    def __init__(self, config: Config):
        self.config = config

        # 初始化各模块
        self.obsidian_connector = ObsidianConnector(
            vault_path=config.OBSIDIAN_VAULT_PATH,
            api_url=config.OBSIDIAN_API_URL,
            api_key=config.OBSIDIAN_API_KEY,
        )

        # 确保环境变量已设置，以便VectorStore可以使用API嵌入

        if config.OLLAMA_API_KEY:
            os.environ.setdefault("OLLAMA_API_KEY", config.OLLAMA_API_KEY)
        if config.OLLAMA_BASE_URL:
            os.environ.setdefault("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL)
        if config.EMBEDDING_MODEL:
            os.environ.setdefault("EMBEDDING_MODEL", config.EMBEDDING_MODEL)

        self.vector_store = VectorStore(persist_path=config.VECTOR_DB_PATH)
        self.retriever = Retriever(self.vector_store, top_k=config.TOP_K)
        self.prompt_engineer = PromptEngineer()

        # 初始化分词器用于计算token
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def index_obsidian_notes(self):
        """
        索引Obsidian笔记到向量库
        """
        logger.info("开始索引Obsidian笔记...")

        # 获取所有笔记列表
        notes = self.obsidian_connector.list_notes()
        logger.info(self.config.OBSIDIAN_VAULT_PATH)
        logger.info(f"找到 {len(notes)} 个笔记")

        documents = []
        for note in notes:
            content = self.obsidian_connector.get_note_content(note["id"])
            if content.strip():
                # 分块处理长文档
                chunks = self._split_document(content, note["title"], note["id"])
                for i, chunk in enumerate(chunks):
                    doc_id = f"{note['id']}_chunk_{i}"
                    documents.append(
                        {
                            "id": doc_id,
                            "content": chunk,
                            "metadata": {
                                "title": note["title"],
                                "path": note["id"],
                                "chunk_index": i,
                            },
                        }
                    )

        # 批量添加到向量库
        if documents:
            self.vector_store.add_documents(documents)
            logger.info(f"完成索引，共添加 {len(documents)} 个文档块")
        else:
            logger.warning("没有找到可索引的文档内容")

    def _split_document(self, content: str, title: str, note_id: str) -> List[str]:
        """
        分割文档为小块
        参考aidev项目中的文档处理逻辑
        """
        # 简单的按长度分割，保留段落完整性
        chunks = []
        paragraphs = content.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            # 估算token数
            para_token_count = len(self.tokenizer.encode(para))
            current_token_count = len(self.tokenizer.encode(current_chunk))

            if (
                current_token_count + para_token_count > self.config.CHUNK_SIZE
                and current_chunk
            ):
                # 当前块已满，保存并开始新块
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # 添加到当前块
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # 添加最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 如果单个段落太长，强制分割
        final_chunks = []
        for chunk in chunks:
            if len(self.tokenizer.encode(chunk)) > self.config.CHUNK_SIZE:
                # 按字符强制分割
                sub_chunks = self._force_split_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _force_split_chunk(self, text: str) -> List[str]:
        """
        强制分割过长的文本块
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.config.CHUNK_SIZE

            if end >= text_length:
                # 如果end超出文本长度，直接添加剩余文本
                chunks.append(text[start:text_length].strip())
                break

            # 尝试在句子边界分割
            found_boundary = False

            # 向前搜索句子边界
            while end > start:
                if end < text_length and text[end] in ".!?。！？\n":
                    end += 1  # 包含边界字符
                    found_boundary = True
                    break
                end -= 1

            # 如果没有找到合适的句子边界，按固定长度分割
            if not found_boundary:
                end = min(start + self.config.CHUNK_SIZE, text_length)

            chunks.append(text[start:end].strip())
            start = end

            # 添加重叠部分
            if self.config.CHUNK_OVERLAP > 0:
                start = max(
                    start - self.config.CHUNK_OVERLAP, start
                )  # 确保不会倒退到start位置之前

        return chunks

    def query(self, user_question: str) -> str:
        """
        处理用户查询
        """
        logger.info(f"处理查询: {user_question}")

        # 1. 检索相关文档
        context = self.retriever.retrieve_and_format(user_question)

        # 2. 构建提示词
        prompt = self.prompt_engineer.build_rag_prompt(user_question, context)

        # 3. 调用Ollama模型
        try:
            response = ollama.chat(
                model=self.config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response["message"]["content"]
            logger.info("成功获取模型回答")
            return answer
        except Exception as e:
            logger.error(f"调用Ollama模型失败: {e}")
            return f"抱歉，处理您的问题时出现错误: {str(e)}"

    def run_cli(self):
        """
        运行命令行交互界面
        """
        print(
            "RAG智能体已启动！输入 'quit' 或 'exit' 退出，输入 'reindex' 重新索引笔记。"
        )
        print("输入 'status' 查看向量库状态。")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    print("再见！")
                    break
                elif user_input.lower() == "reindex":
                    print("正在重新索引笔记...")
                    self.index_obsidian_notes()
                    print("索引完成！")
                    continue
                elif user_input.lower() == "status":
                    stats = self.vector_store.collection.count()
                    print(f"向量库状态: 当前有 {stats} 个文档块")
                    continue
                elif not user_input:
                    continue

                # 处理查询
                answer = self.query(user_input)
                print(f"\n回答: {answer}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.error(f"处理输入时出错: {e}")
                print(f"出现错误: {str(e)}")


def main():
    """
    主函数
    """
    # 加载配置
    config = Config()

    # 检查必要的配置
    if not config.OBSIDIAN_VAULT_PATH:
        print("错误: 请设置 OBSIDIAN_VAULT_PATH 环境变量指向您的Obsidian知识库路径")
        return

    # 创建RAG智能体
    agent = RAGAgent(config)

    # 检查向量库是否为空，如果为空则索引笔记
    try:
        # 尝获取一个简单的统计来判断是否已有数据
        stats = agent.vector_store.collection.count()
        print(f"当前向量库状态: {stats} 个文档块")
        if stats == 0:
            print("检测到向量库为空，正在索引Obsidian笔记...")
            agent.index_obsidian_notes()
            print("索引完成！")
    except Exception as e:
        print(f"检查向量库状态时出错: {e}")
        print("正在索引Obsidian笔记...")
        agent.index_obsidian_notes()
        print("索引完成！")

    # 运行命令行界面
    agent.run_cli()


if __name__ == "__main__":
    main()
