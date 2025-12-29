# RAG智能体Demo

基于Obsidian知识库的本地RAG（检索增强生成）智能体Demo。

## 功能特性

- 连接本地Obsidian知识库
- 自动索引笔记内容到向量数据库
- 基于语义检索的问答系统
- 本地大模型推理（通过Ollama）
- 命令行交互界面

## 技术栈

- Python 3.8+
- FastAPI
- ChromaDB（向量数据库）
- Sentence Transformers（中文嵌入）或Ollama嵌入API
- Ollama（本地大模型）
- Obsidian API（可选）

## 环境准备

### 1. 安装Ollama

请先安装Ollama并确保服务运行：

```bash
# macOS
brew install ollama

# 启动Ollama服务
ollama serve
```

### 2. 下载模型

下载一个适合的模型，例如 `deepseek-coder` 或 `llama3`：

```bash
ollama pull deepseek-coder
# 或者
ollama pull llama3
```

### 3. (推荐) 配置Ollama嵌入模型
为了避免网络问题无法下载sentence-transformers模型，推荐使用Ollama的嵌入模型：

```bash
ollama pull nomic-embed-text
```

### 4. Obsidian配置

如果要使用Obsidian API，需要安装相关插件：

1. 在Obsidian中安装 "MCP: Model Context Protocol" 插件，或
2. 安装 "REST API" 插件

或者直接指定Vault路径，通过文件系统访问。

## 安装依赖

```bash
cd /Users/xiejindong/code/canway/bk-aidev-agent/demo
pip install -r requirements.txt
```

## 配置

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置以下配置：

```env
# Obsidian配置
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
OBSIDIAN_API_KEY=your_obsidian_api_key_if_using_api
OBSIDIAN_API_URL=http://localhost:5136

# Ollama配置
OLLAMA_MODEL=deepseek-coder
OLLAMA_HOST=http://localhost:11434

# (推荐) 如果使用Ollama嵌入模型
OLLAMA_API_KEY=your_ollama_api_key_if_needed
OLLAMA_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text

# 向量数据库配置
VECTOR_DB_PATH=./vector_store

# 分块配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 检索配置
TOP_K=5
```

## 运行

```bash
cd /Users/xiejindong/code/canway/bk-aidev-agent/demo
python -m rag_agent.main
```

## 使用说明

1. 首次运行会自动索引Obsidian笔记
2. 输入问题后按回车，智能体会检索相关文档并生成回答
3. 输入 `reindex` 可重新索引笔记
4. 输入 `quit` 或 `exit` 退出程序

## 项目结构

```
demo/
├── rag_agent/           # RAG智能体模块
│   ├── __init__.py
│   ├── config.py        # 配置管理
│   ├── obsidian_connector.py  # Obsidian连接器
│   ├── vector_store.py  # 向量数据库
│   ├── retriever.py     # 检索器
│   ├── prompt_engineer.py  # 提示工程
│   └── main.py          # 主程序
├── requirements.txt     # 依赖列表
├── README.md           # 本文件
└── .env.example        # 环境变量示例
```

## 与AIDev项目集成

此Demo参考了AIDev项目中的以下能力：
- RAG架构设计
- 向量检索实现
- 提示工程策略
- Agent决策机制

## 网络连接问题解决方案

如果遇到网络连接问题（如无法下载sentence-transformers模型），请使用Ollama嵌入模型：

1. 确保Ollama服务正在运行：
   ```bash
   ollama serve
   ```

2. 下载嵌入模型：
   ```bash
   ollama pull nomic-embed-text
   ```

3. 在.env文件中设置：
   ```env
   OLLAMA_API_KEY=dummy_key  # 任意值即可
   OLLAMA_BASE_URL=http://localhost:11434/v1
   EMBEDDING_MODEL=nomic-embed-text
   ```

4. 这样就不需要从HuggingFace下载模型，而是使用Ollama的嵌入API。

## 自定义扩展

1. **更换嵌入模型**：在 `vector_store.py` 中更换SentenceTransformer模型或配置API模型
2. **调整检索策略**：在 `retriever.py` 中修改检索逻辑
3. **优化提示词**：在 `prompt_engineer.py` 中调整提示模板
4. **添加Web界面**：基于FastAPI构建Web API

## 注意事项

1. 确保Ollama服务已启动且模型已下载
2. Obsidian Vault路径需要有读取权限
3. 向量库文件会存储在指定路径，占用磁盘空间
4. 首次索引可能需要较长时间，取决于笔记数量
5. 如果遇到网络问题，强烈建议使用Ollama嵌入模型

## 故障排除

- 如果无法连接Ollama：检查服务是否运行，模型是否已下载
- 如果无法读取笔记：检查Vault路径和权限设置
- 如果检索效果不佳：尝试调整嵌入模型或检索参数
- 如果遇到网络超时：配置Ollama嵌入模型或使用代理
- 如果出现numpy版本错误：确保使用numpy<2.0版本