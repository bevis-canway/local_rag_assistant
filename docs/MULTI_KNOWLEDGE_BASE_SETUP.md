# 小魔仙RAG智能体 - 多知识库配置指南

## 概述

小魔仙RAG智能体现在支持多知识库功能，允许您配置和管理多个独立的知识库，并在查询时自动跨所有启用的知识库进行检索。

## 配置方式

### 1. 自动配置（推荐）

使用提供的配置脚本快速配置知识库：

```bash
python examples/flexible_multi_kb_setup.py
```

此脚本会：
- 从配置文件或环境变量读取知识库路径
- 扫描指定目录下的所有子目录
- 为每个子目录创建独立的知识库配置
- 保存配置以便后续使用

### 2. 环境变量配置

在 `.env` 文件中添加多知识库配置：

```env
KNOWLEDGE_BASES_CONFIG='[
  {
    "name": "project_docs",
    "type": "obsidian",
    "path": "/Users/xiejindong/Desktop/multi_knowledge_base/rag_local_km_tset",
    "description": "项目文档知识库",
    "enabled": true,
    "vector_store_path": "./vector_store/project_docs"
  },
  {
    "name": "research_docs", 
    "type": "obsidian",
    "path": "/Users/xiejindong/Desktop/multi_knowledge_base/rag_local_km_tset2",
    "description": "研究文档知识库",
    "enabled": true,
    "vector_store_path": "./vector_store/research_docs"
  }
]'
```

### 3. 代码中动态配置

在代码中动态添加知识库：

```python
from rag_agent.knowledge_base_manager import KnowledgeBaseConfig

# 创建知识库配置
kb_config = KnowledgeBaseConfig(
    name="my_kb",
    type="obsidian",  # 或其他类型
    path="/path/to/knowledge/base",
    description="我的知识库",
    enabled=True,
    vector_store_path="./vector_store/my_kb"
)

# 添加到知识库管理器
agent.knowledge_base_manager.add_knowledge_base(kb_config)
```

## 知识库类型支持

目前支持以下知识库类型：
- `obsidian` - Obsidian笔记知识库
- `folder` - 普通文件夹（未来版本支持）
- `database` - 数据库知识库（未来版本支持）

## 索引知识库

配置完成后，需要索引知识库以使其内容可检索：

```bash
# 确保Ollama服务正在运行
ollama serve

# 然后运行索引
python -c "from examples.flexible_multi_kb_setup import index_knowledge_bases; index_knowledge_bases()"
```

## 使用多知识库

配置并索引完成后，您可以像平常一样使用小魔仙RAG智能体。查询时系统会自动：

1. 跨所有启用的知识库检索相关文档
2. 整合来自不同知识库的相关信息
3. 生成综合性的回答
4. 在流式响应中显示来自不同知识库的参考文档

## 管理知识库

### 查看所有知识库

```python
kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
for kb_info in kb_infos:
    status = "启用" if kb_info.enabled else "禁用"
    print(f"- {kb_info.name} ({status}): {kb_info.description}")
    print(f"  路径: {kb_info.path}")
    print(f"  文档数: {kb_info.indexed_documents_count}")
```

### 启用/禁用知识库

```python
# 临时禁用某个知识库
kb = agent.knowledge_base_manager.get_knowledge_base("kb_name")
kb.enabled = False

# 或者删除知识库
agent.knowledge_base_manager.remove_knowledge_base("kb_name")
```

### 保存和加载配置

```python
# 保存配置
agent.knowledge_base_manager.save_configs("kb_configs.json")

# 加载配置
agent.knowledge_base_manager.load_configs("kb_configs.json")
```

## 注意事项

1. **Ollama服务**：确保Ollama服务在索引和查询时正在运行
2. **存储空间**：每个知识库都有独立的向量存储，需要足够的磁盘空间
3. **性能影响**：知识库数量增加会提高查询时间，但提升回答准确性
4. **文档格式**：确保知识库中的文档格式受支持（目前主要支持Markdown格式）

## 故障排除

### 索引失败
- 检查Ollama服务是否运行：`ollama serve`
- 检查知识库路径是否存在且可读
- 检查网络连接（如果使用API嵌入）

### 查询无结果
- 确认知识库已正确索引
- 检查知识库是否已启用
- 验证查询关键词是否与文档内容匹配

## 示例代码

查看以下示例了解如何使用多知识库功能：
- `examples/quick_multi_kb_setup.py` - 快速配置脚本
- `examples/configure_multi_knowledge_bases.py` - 详细配置示例
- `examples/use_multi_knowledge_bases.py` - 使用示例