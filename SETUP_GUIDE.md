# RAG智能体配置和运行指南

## 重要更新：解决网络连接问题

我们已经更新了RAG智能体，现在支持通过Ollama的API来使用嵌入模型，这样就无需从HuggingFace下载模型，解决了在中国大陆访问HuggingFace的网络问题。

## 快速配置步骤

### 1. 启动Ollama服务
```bash
ollama serve
```

### 2. 下载必需的模型
```bash
# 下载用于嵌入的模型
ollama pull nomic-embed-text

# 下载用于对话的模型（可选，使用您喜欢的模型）
ollama pull deepseek-coder
```

### 3. 配置环境变量
确保您的 `.env` 文件包含以下配置：
```
# API嵌入模型配置（关键配置，解决网络问题）
OLLAMA_API_KEY=dummy_key
OLLAMA_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text

# 其他配置...
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
OLLAMA_MODEL=deepseek-coder
```

## 运行RAG智能体

### 方法1：使用启动脚本（推荐）
```bash
./start_rag_agent.sh
```

### 方法2：手动运行
```bash
cd /Users/xiejindong/code/canway/bk-aidev-agent/demo
python -m rag_agent.main
```

## 故障排除

### 1. 如果遇到"Connection error"
- 确保Ollama服务正在运行：`ollama serve`
- 确保已下载nomic-embed-text模型：`ollama pull nomic-embed-text`

### 2. 如果仍然尝试下载HuggingFace模型
- 检查环境变量是否正确设置
- 确认OLLAMA_API_KEY不为空
- 重启Python环境使环境变量生效

### 3. 如果Obsidian路径有中文或空格
- 程序已支持包含中文和空格的路径
- 确保路径使用绝对路径格式

## 验证配置是否成功

运行以下命令验证API嵌入模型是否被正确使用：
```bash
python -c "
import os
os.environ['OLLAMA_API_KEY']='dummy_key'
os.environ['OLLAMA_BASE_URL']='http://localhost:11434/v1'
os.environ['EMBEDDING_MODEL']='nomic-embed-text'
from rag_agent.main import RAGAgent
from rag_agent.config import Config
config = Config()
agent = RAGAgent(config)
print('✓ API嵌入模型配置成功:', agent.vector_store.use_api_embeddings)
"
```

如果输出"✓ API嵌入模型配置成功: True"，说明配置正确。

## 项目架构

- [rag_agent/](./rag_agent/) - RAG智能体核心模块
  - [config.py](./rag_agent/config.py) - 配置管理
  - [obsidian_connector.py](./rag_agent/obsidian_connector.py) - Obsidian连接器
  - [vector_store.py](./rag_agent/vector_store.py) - 向量数据库（支持API嵌入）
  - [retriever.py](./rag_agent/retriever.py) - 检索器
  - [prompt_engineer.py](./rag_agent/prompt_engineer.py) - 提示工程
  - [main.py](./rag_agent/main.py) - 主程序

## 关键改进

1. **API嵌入支持**：现在支持通过Ollama API获取嵌入向量，无需下载本地模型
2. **错误处理**：改进了错误处理机制，提供更清晰的错误信息
3. **文档分割修复**：修复了文档分割中的索引越界问题
4. **环境变量管理**：确保配置正确传递到各组件

现在您可以顺利使用RAG智能体，不再受网络限制影响！