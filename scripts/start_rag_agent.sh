#!/bin/bash
# 小魔仙RAG智能体快速启动脚本

echo "小魔仙RAG智能体快速启动脚本"
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
if ! ollama list | grep -q "bge-m3:latest"; then
    echo "⚠️  bge-m3:latest模型未找到，正在下载..."
    if ollama pull bge-m3:latest; then
        echo "✓ bge-m3:latest模型下载完成"
    else
        echo "⚠️  无法下载bge-m3:latest模型，尝试下载nomic-embed-text作为备选..."
        if ollama pull nomic-embed-text; then
            echo "✓ nomic-embed-text模型下载完成"
        else
            echo "❌ 无法下载必需的嵌入模型"
            exit 1
        fi
    fi
else
    echo "✓ bge-m3:latest模型已存在"
fi

# 检查Python环境
echo "检查Python环境..."
if command -v uv &> /dev/null; then
    echo "✓ uv已安装，使用uv同步依赖..."
    if uv sync; then
        echo "✓ 依赖同步完成"
    else
        echo "❌ 依赖同步失败"
        exit 1
    fi
else
    echo "⚠️  uv未安装，尝试使用pip安装依赖..."
    if [ -f requirements.txt ]; then
        if pip install -r requirements.txt; then
            echo "✓ 依赖安装完成"
        else
            echo "❌ 依赖安装失败"
            exit 1
        fi
    else
        echo "❌ requirements.txt文件不存在"
        exit 1
    fi
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

# 启动小魔仙RAG智能体
echo "启动小魔仙RAG智能体..."
uv run python rag_agent/main.py