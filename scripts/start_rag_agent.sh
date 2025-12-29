#!/bin/bash
# RAG智能体快速启动脚本

echo "RAG智能体快速启动脚本"
echo "========================"

# 检查Ollama是否运行
if ! pgrep -f "ollama serve" > /dev/null && ! curl -s http://localhost:11434 > /dev/null; then
    echo "⚠️  Ollama服务未运行，正在启动..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    if ! curl -s http://localhost:11434 > /dev/null; then
        echo "❌ 无法启动Ollama服务，请手动启动: ollama serve"
        exit 1
    fi
    echo "✓ Ollama服务已启动"
else
    echo "✓ Ollama服务正在运行"
fi

# 检查必需的模型
echo "检查必需的模型..."
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "⚠️  nomic-embed-text模型未找到，正在下载..."
    if ollama pull nomic-embed-text; then
        echo "✓ nomic-embed-text模型下载完成"
    else
        echo "❌ 无法下载nomic-embed-text模型"
        exit 1
    fi
else
    echo "✓ nomic-embed-text模型已存在"
fi

# 检查Python环境
echo "检查Python环境..."
if ! python -c "import chromadb, sentence_transformers, openai, ollama, tiktoken" > /dev/null 2>&1; then
    echo "⚠️  部分依赖未安装，正在安装..."
    if pip install -r requirements.txt; then
        echo "✓ 依赖安装完成"
    else
        echo "❌ 依赖安装失败"
        exit 1
    fi
else
    echo "✓ 所有依赖已安装"
fi

# 检查环境变量
if [ ! -f .env ]; then
    echo "⚠️  .env文件不存在，创建示例文件..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "已创建.env文件，请编辑其中的配置"
    else
        echo "❌ .env.example文件也不存在"
        exit 1
    fi
fi

echo "✓ 环境检查完成"

# 启动RAG智能体
echo "启动RAG智能体..."
python -m rag_agent.main