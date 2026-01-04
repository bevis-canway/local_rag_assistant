# 小魔仙RAG智能体

基于本地模型的 RAG（检索增强生成）智能体，专门用于处理 Obsidian 笔记知识库。

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

## 技术栈

- **语言**: Python 3.11+
- **依赖管理**: uv
- **向量数据库**: ChromaDB
- **嵌入模型**: Ollama (支持 bge-m3:latest 等本地模型)
- **API框架**: FastAPI
- **构建工具**: Makefile
- **代码质量**: Ruff

## 快速开始

### 环境要求

- Python 3.11+
- Ollama (用于本地模型)
- uv (现代 Python 包管理器)

### 安装

1. **克隆项目**
   ```bash
   git clone https://github.com/bevis-canway/local_rag_assistant.git
   cd local_rag_assistant
   ```

2. **安装 uv (如果尚未安装)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **安装项目依赖**
   ```bash
   make install
   ```

4. **启动 Ollama 服务**
   ```bash
   ollama serve
   ```

5. **下载必需的模型**
   ```bash
   ollama pull bge-m3:latest
   ollama pull deepseek-coder
   ```

6. **配置环境变量**
   复制并编辑 `.env` 文件：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件中的配置
   ```

### 运行

1. **使用 Makefile 运行（推荐）**
   ```bash
   make run
   ```

2. **使用启动脚本**
   ```bash
   make start
   ```

3. **开发模式**
   ```bash
   make dev
   ```

## 项目结构

```
local_rag_assistant/
├── rag_agent/              # 小魔仙RAG智能体核心模块
│   ├── config.py          # 配置管理
│   ├── main.py            # 主程序入口
│   ├── obsidian_connector.py # Obsidian连接器
│   ├── prompt_engineer.py # 提示工程
│   ├── retriever.py       # 检索器（含自适应相似度过滤）
│   ├── vector_store.py    # 向量数据库管理
│   ├── hallucination_detector.py # 幻觉检测模块
│   └── prompts/           # 提示词管理
│       ├── __init__.py
│       ├── prompt_templates.py # 提示词模板
│       └── hallucination_templates.py # 幻觉检测提示词
├── scripts/               # 脚本文件
│   └── start_rag_agent.sh # 启动脚本
├── tests/                 # 测试套件
│   └── test_hallucination_detection.py # 幻觉检测测试
├── docs/                  # 文档
├── roadmap/               # 产品路标
├── Makefile               # 自动化构建工具
├── pyproject.toml         # 项目依赖和配置
└── ...
```

## 自动化构建工具

项目集成了 Makefile，提供便捷的命令行工具：

```bash
make help          # 查看所有可用命令
make install       # 安装项目依赖
make run           # 运行小魔仙RAG智能体服务
make dev           # 启动开发模式（热重载）
make test          # 运行测试套件
make lint          # 代码质量检查
make format        # 格式化代码
make reset-db      # 重置向量数据库
make clean         # 清理构建产物
make purge         # 完全清理（包括虚拟环境）
```

## 核心功能详解

### 幻觉检测与预防
小魔仙RAG智能体集成了先进的幻觉检测机制：
- **双重检测机制**：基于事实对比和语义一致性的双重检测
- **优化提示词**：添加防止幻觉的明确指令
- **生成参数优化**：配置较低的temperature（0.2）减少随机性
- **置信度评估**：对回答准确性进行量化评估

### 优化的检索策略
采用自适应相似度过滤策略，解决文档相关但被错误过滤的问题：
- **动态阈值**：基于检索结果的平均相似度调整过滤标准
- **多级过滤**：实现回退机制，确保相关文档不被错误过滤
- **保留相对匹配**：确保即使整体相似度不高但相对匹配度最高的文档也能被保留

### 提示词管理
采用统一的提示词管理机制：
- **集中管理**：所有提示词模板统一管理在 `rag_agent/prompts/` 目录
- **功能分类**：按功能分类管理（RAG相关、系统提示词、幻觉检测等）
- **中文优先**：所有提示词优先使用中文版本
- **可维护性**：提高可维护性和可扩展性

## 技术选型说明

本项目选择当前技术栈而非 LangChain 的原因：
- 更适合本地 RAG 场景，代码更简洁
- 对本地模型（Ollama）有更好的支持
- 更轻量级，避免框架开销
- 更直接的控制和调试能力

详细技术选型对比请参见 [docs/tech_comparison.md](docs/tech_comparison.md)

## 提示词管理

本项目采用统一的提示词管理机制：
- 所有提示词模板集中管理在 `rag_agent/prompts/` 目录
- 按功能分类管理（RAG相关、系统提示词、幻觉检测等）
- 提高可维护性和可扩展性

详细说明请参见 [docs/prompt_management.md](docs/prompt_management.md)

## 幻觉缓解策略

本项目实现多种策略来减少大模型幻觉现象：
- 幻觉检测模块：检测生成回答与文档的一致性
- 优化提示词模板：添加防止幻觉的指令
- 生成参数优化：配置较低temperature减少随机性
- 事实验证机制：对RAG生成的回答进行验证

详细说明请参见 [docs/HALLUCINATION_PREVENTION.md](docs/HALLUCINATION_PREVENTION.md)

## 产品路标

了解项目的发展方向和未来规划：
- [功能迭代历史](roadmap/feature_iterations.md) - 查看已完成的功能迭代
- [待办功能列表](roadmap/todo_features.md) - 查看待实现的功能和优化
- [产品路标总览](roadmap/roadmap_overview.md) - 查看长期发展规划

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

[在此处添加许可证信息]