# justfile
set shell := ["bash", "-c"]

# 默认显示帮助
default: help

# 显示所有命令
help:
    @just --list --unsorted

# 清理
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf *.egg-info .pytest_cache .ruff_cache .cache logs

# 测试
test:
    pytest tests/

# 代码检查与格式化
lint:
    ruff check .

# 自动格式化代码
format:
	ruff format .

# 自动修复代码问题, 优化引入并格式化
# 前面加上-号, 表示执行命令时, 错误了也继续执行后续命令
fix:
    -ruff check --fix .
    ruff check --select I --fix .
    ruff format .

run:
    uv run python src/contextos/main.py