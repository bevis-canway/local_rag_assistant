# 小魔仙RAG智能体提示词管理说明

## 概述

小魔仙RAG智能体采用统一的提示词管理机制，将所有提示词模板集中管理，以提高可维护性和可扩展性。

## 目录结构

```
rag_agent/
├── prompts/
│   ├── __init__.py
│   └── prompt_templates.py
├── retriever.py
├── prompt_engineer.py
└── ...
```

## 提示词分类

### RAG相关提示词 (RAG_PROMPT_TEMPLATES)

包含以下模板：

1. **no_document_found** - 当未找到相关文档时的提示词
   - 用于在检索不到相关文档时，调用本地大模型进行回答
   - 包含对用户问题的引用和指导模型基于通用知识回答的说明

2. **rag_answer** - RAG问答提示词
   - 用于基于检索到的上下文信息回答用户问题
   - 包含详细的上下文处理和回答指导

### 系统提示词 (SYSTEM_PROMPTS)

包含系统级的提示词模板：

1. **default_assistant** - 默认助手提示词
2. **rag_assistant** - RAG助手专用提示词

### 格式化提示词 (FORMATTING_PROMPTS)

包含回答格式相关的提示词模板：

1. **response_format** - 回答格式指导

## 使用方法

### 在模块中引入

```python
from rag_agent.prompts.prompt_templates import RAG_PROMPT_TEMPLATES, SYSTEM_PROMPTS
```

### 使用模板

```python
# 使用RAG相关提示词
prompt = RAG_PROMPT_TEMPLATES["rag_answer"].format(context=context, query=query)

# 使用系统提示词
system_prompt = SYSTEM_PROMPTS["rag_assistant"]
```

## 扩展指南

### 添加新的提示词模板

1. 在对应的字典中添加新的键值对：

```python
RAG_PROMPT_TEMPLATES = {
    # ... 现有模板 ...
    "new_template_name": "新的提示词模板内容 {placeholder}",
}
```

2. 确保模板中的占位符使用 `{variable_name}` 格式

### 更新现有模板

直接修改对应模板的值，注意保持占位符不变。

## 最佳实践

1. **保持一致性** - 所有相似功能的提示词应保持风格一致
2. **使用占位符** - 为动态内容使用 `{variable_name}` 占位符
3. **清晰命名** - 模板键名应清晰表达其用途
4. **文档注释** - 为复杂的提示词添加注释说明
5. **测试验证** - 更新提示词后应测试其效果

## 维护说明

- 定期审查和优化提示词效果
- 根据用户反馈调整提示词内容
- 保持提示词的简洁性和有效性
- 确保所有占位符都被正确使用