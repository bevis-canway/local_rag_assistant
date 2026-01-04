# 小魔仙RAG智能体配置和运行指南

## 项目概述

小魔仙RAG智能体是一个现代化的检索增强生成系统，使用 uv 作为依赖管理工具，支持通过 Ollama API 使用本地模型进行嵌入计算。项目已集成完整的自动化构建工具，简化开发和部署流程。

## 核心特性

- **本地模型支持**：优先使用名为'小魔仙'的本地模型（对应 bge-m3:latest）进行嵌入计算
- **Obsidian 集成**：直接连接 Obsidian 知识库，支持 Markdown 格式
- **现代化依赖管理**：使用 uv 替代传统 pip，提供更快的依赖解析和安装
- **自动化构建工具**：集成 Makefile，提供一键式开发体验
- **代码质量保障**：集成 Ruff 进行代码格式化和质量检查
- **提示词管理**：统一管理所有提示词模板，提高可维护性
- **幻觉检测与预防**：集成双重检测机制，减少大模型幻觉现象
- **优化的检索策略**：采用自适应相似度过滤，提高文档检索准确性
- **流式响应功能**：支持实时传输AI生成内容，用户可查看AI思考过程

## 环境要求

- Python 3.11+
- Ollama (用于本地模型)
- uv (现代 Python 包管理器)

### 安装 uv (如果尚未安装)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 快速开始

### 1. 克隆并初始化项目
```bash
git clone https://github.com/bevis-canway/local_rag_assistant.git
cd local_rag_assistant

# 安装项目依赖
make install
```

### 2. 启动Ollama服务
```bash
ollama serve
```

### 3. 下载必需的模型
```bash
# 下载用于嵌入的模型（推荐使用小魔仙模型）
ollama pull bge-m3:latest

# 下载用于对话的模型（可选，使用您喜欢的模型）
ollama pull deepseek-coder
```

### 4. 配置环境变量
复制并编辑 `.env` 文件：
```
# 小魔仙模型配置
OLLAMA_API_KEY=dummy_key
OLLAMA_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=bge-m3:latest

# Obsidian配置
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
OBSIDIAN_API_URL=http://localhost:5136
OBSIDIAN_API_KEY=your_obsidian_api_key

# Ollama配置
OLLAMA_MODEL=deepseek-coder
OLLAMA_HOST=http://localhost:11434

# 向量数据库配置
VECTOR_DB_PATH=./vector_store

# 其他配置
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
```

## 使用自动化构建工具

项目集成了 Makefile，提供便捷的命令行工具：

### 查看所有可用命令
```bash
make help
```

### 主要命令
```bash
# 安装项目依赖
make install

# 运行小魔仙RAG智能体服务
make run

# 启动开发模式（热重载）
make dev

# 运行测试套件
make test

# 代码质量检查
make lint

# 格式化代码
make format

# 检查依赖安全和兼容性
make check

# 重置向量数据库
make reset-db

# 完全清理（包括虚拟环境）
make purge
```

## 运行小魔仙RAG智能体

### 方法1：使用Makefile（推荐）
```bash
make run
```

### 方法2：使用uv直接运行
```bash
uv run python rag_agent/main.py
```

### 方法3：启动开发模式
```bash
make dev
```

### 方法4：使用启动脚本
```bash
bash scripts/start_rag_agent.sh
```

## 配置小魔仙模型

### 使用预设命令配置小魔仙模型
```bash
make init-model
```

### 手动验证小魔仙模型
```bash
uv run python -c "import ollama; ollama.embeddings(model='bge-m3:latest', prompt='小魔仙模型测试'); print('小魔仙模型已准备就绪')"
```

## 项目架构

- [rag_agent/](./rag_agent/) - 小魔仙RAG智能体核心模块
  - [config.py](./rag_agent/config.py) - 配置管理
  - [obsidian_connector.py](./rag_agent/obsidian_connector.py) - Obsidian连接器
  - [vector_store.py](./rag_agent/vector_store.py) - 向量数据库（支持API嵌入）
  - [retriever.py](./rag_agent/retriever.py) - 检索器（含自适应相似度过滤）
  - [prompt_engineer.py](./rag_agent/prompt_engineer.py) - 提示工程
  - [main.py](./rag_agent/main.py) - 主程序
  - [hallucination_detector.py](./rag_agent/hallucination_detector.py) - 幻觉检测模块
  - [prompts/](./rag_agent/prompts/) - 提示词管理
    - [prompt_templates.py](./rag_agent/prompts/prompt_templates.py) - RAG相关提示词
    - [hallucination_templates.py](./rag_agent/prompts/hallucination_templates.py) - 幻觉检测相关提示词

- [tests/](./tests/) - 测试套件
  - [test_connection.py](./tests/test_connection.py) - 连接测试
  - [test_demo.py](./tests/test_demo.py) - 演示测试
  - [test_embeddings.py](./tests/test_embeddings.py) - 嵌入测试
  - [test_hallucination_detection.py](./tests/test_hallucination_detection.py) - 幻觉检测测试

- [scripts/](./scripts/) - 脚本文件
  - [start_rag_agent.sh](./scripts/start_rag_agent.sh) - 启动脚本

- [Makefile](./Makefile) - 自动化构建工具
- [pyproject.toml](./pyproject.toml) - 项目依赖和配置
- [uv.lock](./uv.lock) - 依赖锁定文件

## 故障排除

### 1. 如果遇到"Connection error"
- 确保Ollama服务正在运行：`ollama serve`
- 确保已下载bge-m3:latest模型：`ollama pull bge-m3:latest`

### 2. 如果依赖安装失败
```bash
# 重新安装依赖
make install

# 或完全清理后重新安装
make purge
make install
```

### 3. 如果向量数据库出错
```bash
# 重置向量数据库
make reset-db
```

### 4. 如果环境变量未生效
```bash
# 重新同步虚拟环境
make sync
```

## 验证配置是否成功

运行以下命令验证小魔仙模型是否被正确配置：
```bash
make init-model
```

或手动验证：
```bash
uv run python -c "
import os
os.environ['OLLAMA_API_KEY']='dummy_key'
os.environ['OLLAMA_BASE_URL']='http://localhost:11434/v1'
os.environ['EMBEDDING_MODEL']='bge-m3:latest'
from rag_agent.main import RAGAgent
from rag_agent.config import Config
config = Config()
agent = RAGAgent(config)
print('✓ 小魔仙模型配置成功:', agent.vector_store.use_api_embeddings)
"
```

## 关键特性详解

### 幻觉检测与预防
- **双重检测机制**：基于事实对比和语义一致性的双重检测
- **优化提示词**：添加防止幻觉的明确指令
- **生成参数优化**：配置较低的temperature（0.2）减少随机性
- **置信度评估**：对回答准确性进行量化评估

### 优化的检索策略
- **自适应阈值**：基于检索结果的平均相似度动态调整过滤标准
- **多级过滤**：实现回退机制，确保相关文档不被错误过滤
- **保留相对匹配**：确保即使整体相似度不高但相对匹配度最高的文档也能被保留

### 提示词管理
- **集中管理**：所有提示词模板统一管理
- **功能分类**：按RAG、系统、格式化等分类管理
- **中文优先**：所有提示词优先使用中文版本

现在您可以使用现代化的工具链顺利运行小魔仙RAG智能体！