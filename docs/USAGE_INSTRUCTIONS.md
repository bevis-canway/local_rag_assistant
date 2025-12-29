# RAG智能体使用说明

## 本地模型配置

您的系统已配置为使用本地Ollama模型：

- **对话模型**: `deepseek-r1:14b` - 用于回答用户问题
- **嵌入模型**: `bge-m3:latest` - 用于向量化文档和查询

## 启动步骤

### 1. 确保Ollama服务正在运行
通常Ollama服务会在后台自动运行，您可以通过以下命令验证：
```bash
ollama list
```

### 2. 启动RAG智能体
```bash
cd /Users/xiejindong/code/canway/bk-aidev-agent/demo
python -m rag_agent.main
```

## 首次运行

首次运行时，系统会：
1. 自动索引您的193个Obsidian笔记
2. 使用`bge-m3:latest`模型为每个文档块生成向量表示
3. 将向量存储在本地ChromaDB数据库中

## 使用命令

在RAG智能体运行后，您可以使用以下命令：

- **普通查询**: 直接输入问题，智能体会从Obsidian笔记中检索相关信息并回答
- **`reindex`**: 重新索引所有Obsidian笔记（当您添加了新笔记后使用）
- **`status`**: 查看向量库状态
- **`quit` 或 `exit`**: 退出程序

## 模型优势

### bge-m3:latest
- 支持多语言，包括中文
- 适合长文档的向量化
- 本地运行，无需网络连接

### deepseek-r1:14b
- 14B参数的大模型
- 支持复杂推理任务
- 适合中文对话

## 性能优化

- **向量数据库**: 使用ChromaDB进行高效相似性搜索
- **文档分块**: 自动将长文档分割为1000 token的块
- **上下文窗口**: 根据模型能力动态调整上下文长度

## 故障排除

### 如果遇到连接问题
1. 检查Ollama服务是否运行：`ollama list`
2. 如果服务未运行，启动它：`ollama serve`
3. 确保所需模型已下载：`ollama pull bge-m3:latest` 和 `ollama pull deepseek-r1:14b`

### 如果索引过程太慢
- 系统需要处理193个笔记，可能需要几分钟
- 请耐心等待索引完成
- 索引完成后，后续查询将非常快速

## 自定义配置

如需更改配置，请编辑 `.env` 文件：
- `OLLAMA_MODEL`: 更改对话模型
- `EMBEDDING_MODEL`: 更改嵌入模型
- `CHUNK_SIZE`: 调整文档块大小
- `TOP_K`: 调整检索的文档数量

现在您可以享受完全本地化的RAG智能体体验，所有处理都在您的机器上完成，无需网络连接！