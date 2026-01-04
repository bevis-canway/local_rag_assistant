"""
流式响应处理模块
为小魔仙RAG智能体提供流式事件处理能力
参考aidev项目实现流式响应功能
"""
import asyncio
import json
import enum
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


class EventType(enum.Enum):
    """事件类型枚举"""
    LOADING = "loading"
    TEXT = "text"          # 普通文本内容
    DONE = "done"          # 完成事件
    ERROR = "error"        # 错误事件
    REFERENCE_DOC = "reference_doc"  # 参考文档
    THINK = "think"        # 思考过程


@dataclass
class StreamEvent:
    """流式事件数据类"""
    event: EventType
    content: str = ""
    cover: bool = False  # 是否覆盖显示
    documents: Optional[list] = None
    elapsed_time: Optional[float] = None  # 思考时间
    metadata: Optional[Dict[str, Any]] = None


class StreamingHandler:
    """
    流式响应处理器
    实现类似aidev项目的流式事件处理能力
    """
    
    def __init__(self):
        self.start_time = None
    
    async def generate_stream_response(
        self,
        user_question: str,
        get_response_func,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        生成流式响应
        """
        # 开始思考过程
        self.start_time = datetime.now()
        
        # 发送开始事件
        yield StreamEvent(
            event=EventType.LOADING,
            content="正在处理您的问题...",
            cover=False
        )
        
        try:
            # 发送思考事件
            yield StreamEvent(
                event=EventType.THINK,
                content="正在分析您的问题意图...",
                cover=False
            )
            
            # 获取意图识别结果
            intent_result = kwargs.get('intent_result')
            if intent_result:
                yield StreamEvent(
                    event=EventType.THINK,
                    content=f"识别到意图类型: {intent_result.intent_type}，置信度: {intent_result.confidence}",
                    cover=False
                )
            
            # 如果是知识查询，发送检索事件
            if intent_result and intent_result.intent_type in ["knowledge_query", "ambiguous"]:
                yield StreamEvent(
                    event=EventType.THINK,
                    content="正在检索相关文档...",
                    cover=False
                )
                
                # 获取检索结果
                filtered_results = kwargs.get('filtered_results')
                if filtered_results:
                    yield StreamEvent(
                        event=EventType.REFERENCE_DOC,
                        content=f"找到 {len(filtered_results)} 个相关文档",
                        documents=filtered_results,
                        cover=False
                    )
            
            # 发送生成回答事件
            yield StreamEvent(
                event=EventType.THINK,
                content="正在生成回答...",
                cover=False
            )
            
            # 获取最终回答
            response_text = await get_response_func()
            
            # 发送文本内容（流式）
            # 模拟流式发送文本内容
            for i in range(0, len(response_text), 10):  # 每10个字符发送一次
                chunk = response_text[i:i+10]
                yield StreamEvent(
                    event=EventType.TEXT,
                    content=chunk,
                    cover=False
                )
                await asyncio.sleep(0.01)  # 模拟流式延迟
            
            # 发送完成事件
            elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            yield StreamEvent(
                event=EventType.DONE,
                content="回答生成完成",
                elapsed_time=elapsed
            )
            
        except Exception as e:
            yield StreamEvent(
                event=EventType.ERROR,
                content=f"处理过程中出现错误: {str(e)}",
                cover=False
            )
    
    async def generate_ollama_stream_response(
        self,
        prompt: str,
        model: str,
        options: Dict = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        生成Ollama模型的流式响应
        """
        try:
            # 发送开始事件
            yield StreamEvent(
                event=EventType.LOADING,
                content="正在调用AI模型...",
                cover=False
            )
            
            # 调用Ollama模型进行流式生成
            stream = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options=options or {},
                    stream=True
                )
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.get('message') and chunk['message'].get('content'):
                    content = chunk['message']['content']
                    full_response += content
                    
                    yield StreamEvent(
                        event=EventType.TEXT,
                        content=content,
                        cover=False
                    )
            
            # 发送完成事件
            yield StreamEvent(
                event=EventType.DONE,
                content="AI回答生成完成",
                cover=False
            )
            
        except Exception as e:
            yield StreamEvent(
                event=EventType.ERROR,
                content=f"AI模型调用失败: {str(e)}",
                cover=False
            )
    
    def format_sse_event(self, event: StreamEvent) -> str:
        """
        格式化为Server-Sent Events格式
        """
        data = {
            "event": event.event.value,
            "content": event.content,
            "cover": event.cover,
        }
        
        if event.documents is not None:
            data["documents"] = event.documents
        if event.elapsed_time is not None:
            data["elapsed_time"] = event.elapsed_time
        if event.metadata is not None:
            data["metadata"] = event.metadata
            
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# 为了兼容现有代码，我们还需要导入ollama
import ollama