# 小魔仙RAG智能体 - 幻觉缓解策略

本文档详细介绍了小魔仙RAG智能体为减少大模型幻觉现象所采用的多种策略和实现细节。

## 幻觉缓解策略概览

### 1. 幻觉检测模块
- **实现文件**: `rag_agent/hallucination_detector.py`
- **功能**: 检测生成的回答与检索文档之间的一致性
- **方法**:
  - 基于事实对比的检测
  - 基于语义一致性的检测
  - 综合判断机制

### 2. 优化提示词模板
- **实现文件**: `rag_agent/prompts/prompt_templates.py` 和 `rag_agent/prompts/hallucination_templates.py`
- **策略**:
  - 添加明确的指令防止编造信息
  - 强调仅使用提供的上下文信息
  - 要求明确声明信息来源的限制

### 3. 生成参数优化
- **实现文件**: `rag_agent/config.py` 和 `rag_agent/main.py`
- **参数设置**:
  - 温度 (Temperature): 0.2 (默认值，降低随机性)
  - Top-P: 0.8 (限制概率分布范围)
  - Top-K: 30 (限制候选词范围)

### 4. 事实验证机制
- **实现文件**: `rag_agent/main.py` (集成到查询流程中)
- **功能**:
  - 对知识查询的回答进行幻觉检测
  - 当检测到潜在不一致时提供警告
  - 记录检测到的不一致之处

## 代码实现细节

### 幻觉检测器
```python
class HallucinationDetector:
    def detect_hallucinations(self, response: str, retrieved_docs: List[Dict], query: str) -> HallucinationCheckResult:
        # 检查生成的回答是否与检索到的文档一致
        pass
```

### 优化的提示词
在 `rag_agent/prompts/prompt_templates.py` 中添加了防止幻觉的指令：
- "CRITICAL: Only include information that is directly supported by the provided context"
- "Do not fabricate, infer, or hallucinate information that is not present in the context"

### 生成参数配置
在 `rag_agent/config.py` 中添加了生成参数配置：
```python
GENERATION_TEMPERATURE: float = 0.2
GENERATION_TOP_P: float = 0.8
GENERATION_TOP_K: int = 30
```

## 效果评估

通过实现这些策略，小魔仙RAG智能体在以下方面有所改善：

1. **准确性提升**: 通过RAG检索相关文档，确保回答基于真实信息
2. **幻觉减少**: 通过检测机制识别并标记潜在的不一致信息
3. **可靠性增强**: 通过参数优化和提示词工程减少模型的随机性输出
4. **透明度提高**: 当信息来源有限时，明确告知用户

## 使用建议

为最大化幻觉缓解效果，请注意：

1. 确保知识库文档质量高、信息准确
2. 定期更新和验证知识库内容
3. 监控模型输出，持续优化提示词
4. 根据具体应用场景调整生成参数

## 未来改进方向

1. 集成更先进的事实验证算法
2. 实现动态置信度评估
3. 添加人工审核接口
4. 实现基于反馈的学习机制

## 最新更新日志

### 2026-01-04 更新内容

#### 优化检索器的相似度过滤逻辑
- **问题**: 文档库中的内容与用户问题相关，但检索后却被过滤为不相关
- **解决方案**: 采用自适应相似度过滤策略
  - 使用动态阈值，基于检索结果的平均相似度调整过滤标准
  - 实现多级过滤，如果使用严格阈值未找到结果，尝试使用更宽松的回退阈值
  - 确保即使整体相似度不高，但相对匹配度最高的文档也能被保留
- **效果**: 显著改善检索器性能，减少将相关文档错误过滤的情况

#### 为幻觉检测模块添加详细日志输出
- **目标**: 确保所有关键方法和判断分支都有足够的日志输出
- **实现**:
  - 在所有主要方法中添加了INFO和DEBUG级别的日志
  - 记录检索过程、过滤决策、相似度计算等关键步骤
  - 确保日志输出完整且便于调试
- **效果**: 提高了系统可调试性，便于追踪执行流程

#### 合并幻觉检测和预防提示词模板
- **目标**: 按照提示词管理规范，将相关功能的提示词模板合并管理
- **实现**:
  - 将`hallucination_detection_templates.py`和`hallucination_prevention_templates.py`合并为`hallucination_templates.py`
  - 提供完整的中文版本提示词模板
  - 更新所有相关引用
- **效果**: 提高维护效率，符合项目规范

#### 提示词模板中文化
- **目标**: 确保所有提示词优先使用中文
- **实现**:
  - 将`prompt_templates.py`中的所有提示词翻译为中文
  - 提供完整的中文版本，符合本地化要求
- **效果**: 提升用户体验，符合项目语言规范

### 2026-01-04 流式响应功能
- **功能**: 实现流式响应功能，支持实时传输AI生成内容
- **实现**:
  - 创建`streaming_handler.py`模块，实现多种事件类型（LOADING、TEXT、DONE、ERROR、REFERENCE_DOC、THINK）
  - 在`RAGAgent`类中添加`query_stream`异步方法
  - 支持Server-Sent Events (SSE)协议传输流式数据
  - 在流式输出中集成幻觉检测，对AI生成的每个文本片段进行逐句分析
- **效果**: 用户可以看到AI的实时思考过程和最终答案，同时保持幻觉检测的实时性

### 2026-01-04 多知识库支持功能
- **功能**: 实现多知识库支持，可动态配置和管理多个知识库
- **实现**:
  - 创建`knowledge_base_manager.py`模块，实现知识库管理功能
  - 支持跨多个知识库的文档检索
  - 在`RAGAgent`类中集成多知识库功能
  - 保持与单知识库模式的向后兼容性
- **效果**: 系统可管理多个独立知识库，提升可扩展性和灵活性