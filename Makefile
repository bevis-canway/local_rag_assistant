# 小魔仙RAG智能体项目自动化构建工具
.PHONY: help install update sync run dev clean purge check test lint format reset-db

# 默认目标
.DEFAULT_GOAL := help

# 显示帮助信息
help:
	@echo "小魔仙RAG智能体 - 自动化构建工具"
	@echo ""
	@echo "使用方法:"
	@echo "  make install          安装项目依赖"
	@echo "  make update           更新依赖到最新版本"
	@echo "  make sync             同步虚拟环境到 pyproject.toml"
	@echo "  make run              运行 RAG 智能体服务"
	@echo "  make dev              启动开发模式（热重载）"
	@echo "  make test             运行测试套件"
	@echo "  make lint             代码质量检查"
	@echo "  make lint-strict      严格代码质量检查"
	@echo "  make format           格式化代码"
	@echo "  make clean            清理构建产物和缓存"
	@echo "  make purge            完全清理（包括虚拟环境）"
	@echo "  make check            检查依赖安全和兼容性"
	@echo "  make reset-db         重置向量数据库"
	@echo ""

# 安装项目依赖
install:
	@echo "正在安装项目依赖..."
	uv sync
	@echo "依赖安装完成！"

# 更新依赖到最新版本
update:
	@echo "正在更新项目依赖..."
	uv update
	@echo "依赖更新完成！"

# 同步虚拟环境到 pyproject.toml 的状态
sync:
	@echo "正在同步虚拟环境..."
	uv sync
	@echo "虚拟环境同步完成！"

# 运行 RAG 智能体服务
run:
	@echo "启动小魔仙RAG智能体服务..."
	uv run python rag_agent/main.py

# 启动开发模式（热重载）
dev:
	@echo "启动开发模式（热重载）..."
	uv run uvicorn rag_agent.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
test:
	@echo "运行测试套件..."
	uv run python -m pytest tests/ -v

# 代码质量检查（忽略导入顺序问题，因为有些是必需的）
lint:
	@echo "执行代码质量检查（忽略导入顺序问题）..."
	uv run ruff check . --ignore E402

# 严格代码质量检查
lint-strict:
	@echo "执行严格代码质量检查..."
	uv run ruff check .

# 格式化代码
format:
	@echo "格式化代码..."
	uv run ruff format .
	uv run ruff check . --fix --ignore E402

# 清理构建产物和缓存
clean:
	@echo "清理构建产物和缓存..."
	rm -rf .pytest_cache/ .ruff_cache/ __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .coverage coverage.xml htmlcov/
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "清理完成！"

# 完全清理（包括虚拟环境）
purge: clean
	@echo "完全清理项目（包括虚拟环境）..."
	rm -rf .venv/
	@echo "虚拟环境已删除！"

# 检查依赖安全和兼容性
check:
	@echo "检查依赖安全和兼容性..."
	uv sync
	uv run ruff check . --ignore E402
	uv run python -m pip check
	@echo "检查完成！"

# 为小魔仙模型准备的特殊命令
init-model:
	@echo "初始化小魔仙本地模型..."
	@echo "请确保 Ollama 已安装并运行"
	uv run python -c "import ollama; ollama.embeddings(model='bge-m3:latest', prompt='test'); print('小魔仙模型已准备就绪')"

# 启动完整的 RAG 服务
start:
	@echo "启动完整的RAG服务..."
	bash start_rag_agent.sh

# 重置向量数据库
reset-db:
	@echo "重置向量数据库..."
	rm -rf vector_store/ test_vector_store/
	@echo "向量数据库已重置！"