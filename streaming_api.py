#!/usr/bin/env python3
"""
流式响应API端点
演示小魔仙RAG智能体的流式响应功能
"""
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_agent.config import Config
from rag_agent.main import RAGAgent
from rag_agent.streaming_handler import EventType


app = FastAPI(title="小魔仙RAG智能体流式API", description="支持流式响应的RAG智能体API")

# 初始化智能体
config = Config()
agent = RAGAgent(config)


@app.get("/")
def read_root():
    return {"message": "小魔仙RAG智能体流式API", "status": "running"}


@app.get("/stream_query/{query}")
async def stream_query(query: str):
    """
    流式查询端点，返回Server-Sent Events格式的响应
    """
    async def event_generator():
        try:
            async for event in agent.query_stream(query):
                # 根据事件类型生成SSE格式数据
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
                
                # 生成SSE格式的响应
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_data = {
                "event": "error",
                "content": f"处理请求时出现错误: {str(e)}",
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/query/{query}")
async def query(query: str):
    """
    传统查询端点，返回完整响应
    """
    try:
        response = agent.query(query)
        return {"query": query, "response": response}
    except Exception as e:
        return {"query": query, "error": str(e)}


if __name__ == "__main__":
    print("启动小魔仙RAG智能体流式API...")
    print("API端点:")
    print("  - GET / - API状态")
    print("  - GET /query/{query} - 传统查询")
    print("  - GET /stream_query/{query} - 流式查询（支持SSE）")
    print("\n示例:")
    print("  - http://localhost:8000/query/你好")
    print("  - http://localhost:8000/stream_query/你好")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)