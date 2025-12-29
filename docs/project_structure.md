# 小魔仙RAG智能体项目结构说明

## 项目整体架构

```
local_rag_assistant/
├── rag_agent/              # 小魔仙RAG智能体核心模块
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── main.py            # 主程序入口
│   ├── obsidian_connector.py # Obsidian连接器
│   ├── prompt_engineer.py # 提示工程
│   ├── retriever.py       # 检索器
│   └── vector_store.py    # 向量数据库管理
├── scripts/               # 脚本文件
│   └── start_rag_agent.sh # 启动脚本
├── tests/                 # 测试套件
│   ├── __init__.py
│   ├── test_connection.py # 连接测试
│   ├── test_demo.py       # 演示测试
│   └── test_embeddings.py # 嵌入测试
├── docs/                  # 文档
│   ├── tech_comparison.md # 技术选型对比
│   └── project_structure.md # 项目结构说明
├── Makefile               # 自动化构建工具
├── pyproject.toml         # 项目依赖和配置
├── uv.lock                # 依赖锁定文件
├── .gitignore            # Git忽略配置
├── .env.example          # 环境变量示例
├── SETUP_GUIDE.md        # 设置指南
├── USAGE_INSTRUCTIONS.md # 使用说明
├── README.md             # 项目说明
└── requirements.txt      # 依赖列表（兼容传统方式）
```

## 核心模块说明

### rag_agent/ 模块

#### config.py
- 管理所有环境变量和配置项
- 支持 Obsidian、Ollama、向量数据库等配置
- 使用 Pydantic 进行配置验证

#### obsidian_connector.py
- 连接 Obsidian API
- 获取笔记列表和内容
- 支持中文路径和特殊字符处理

#### vector_store.py
- 管理 ChromaDB 向量数据库
- 支持 API 嵌入（Ollama）和本地嵌入
- 处理文档的添加、检索和持久化

#### retriever.py
- 实现文档检索逻辑
- 格式化检索结果
- 与向量存储模块紧密协作

#### prompt_engineer.py
- 构建 RAG 提示词
- 优化提示词结构
- 提高模型回答质量

#### main.py
- 小魔仙 RAG 智能体主类
- 整合所有模块
- 提供命令行界面

### scripts/ 脚本

#### start_rag_agent.sh
- 自动检查 Ollama 服务
- 验证必需模型
- 使用 uv 同步依赖
- 启动小魔仙 RAG 智能体

### tests/ 测试

#### test_connection.py
- 测试 Ollama 连接
- 验证 API 嵌入功能
- 检查 Obsidian 连接

#### test_demo.py
- 演示各模块功能
- 集成测试

#### test_embeddings.py
- 测试 API 嵌入模型
- 测试本地嵌入模型

## 构建和开发流程

### 自动化构建工具 (Makefile)

```bash
make help          # 查看所有可用命令
make install       # 安装项目依赖
make run           # 运行小魔仙RAG智能体
make dev           # 开发模式（热重载）
make test          # 运行测试套件
make lint          # 代码质量检查
make format        # 代码格式化
make reset-db      # 重置向量数据库
make clean         # 清理构建产物
make purge         # 完全清理（包括虚拟环境）
```

### 依赖管理

- 使用 `uv` 作为现代依赖管理工具
- `pyproject.toml` 定义项目依赖
- `uv.lock` 锁定依赖版本
- 支持开发依赖和生产依赖分离

### 配置管理

- 环境变量通过 `.env` 文件管理
- 支持多种嵌入模型配置
- 可配置 Obsidian、Ollama、向量数据库参数

## 部署和运行

### 环境要求

- Python 3.11+
- Ollama (用于本地模型)
- uv (现代 Python 包管理器)

### 运行方式

1. **命令行方式**：
   ```bash
   make run
   ```

2. **脚本方式**：
   ```bash
   make start
   ```

3. **开发模式**：
   ```bash
   make dev
   ```

## 最佳实践

1. **代码质量**：使用 `make format` 和 `make lint` 保持代码风格一致
2. **测试覆盖**：添加新功能时编写相应测试
3. **依赖管理**：使用 `uv` 管理依赖，避免直接使用 pip
4. **配置管理**：敏感信息通过环境变量配置
5. **向量数据库**：使用 `make reset-db` 重置数据库状态