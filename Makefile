.PHONY: help setup install lint typecheck test test-unit test-integration \
        test-e2e coverage run-infra run-all stop clean seed train

PYTHON  := python3.12
UV      := uv
COMPOSE := docker compose
SRC_DIR := src/faultscope
TEST_DIR := tests

##@ General

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} \
	/^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
	/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup

setup: ## Bootstrap development environment
	$(UV) venv --python $(PYTHON)
	$(UV) pip install -e ".[dev,lint]"
	$(UV) run pre-commit install
	@echo "✓ Development environment ready"

install: ## Install production dependencies only
	$(UV) pip install -e .

##@ Code Quality

lint: ## Run ruff linter and formatter check
	$(UV) run ruff check $(SRC_DIR) $(TEST_DIR)
	$(UV) run ruff format --check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Auto-fix lint issues
	$(UV) run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	$(UV) run ruff format $(SRC_DIR) $(TEST_DIR)

typecheck: ## Run mypy static type analysis
	$(UV) run mypy $(SRC_DIR)

##@ Testing

test: ## Run full test suite (unit + integration)
	$(UV) run pytest $(TEST_DIR)/unit $(TEST_DIR)/integration -x

test-unit: ## Run unit tests only (no containers needed)
	$(UV) run pytest $(TEST_DIR)/unit -m unit -v

test-integration: ## Run integration tests (requires Docker)
	$(UV) run pytest $(TEST_DIR)/integration -m integration -v

test-e2e: ## Run end-to-end tests (full stack must be running)
	$(UV) run pytest $(TEST_DIR)/e2e -m e2e -v

coverage: ## Generate HTML coverage report
	$(UV) run pytest $(TEST_DIR)/unit $(TEST_DIR)/integration \
		--cov=$(SRC_DIR) \
		--cov-report=html:reports/coverage \
		--cov-report=term-missing
	@echo "✓ Report at reports/coverage/index.html"

##@ Infrastructure

run-infra: ## Start Kafka, TimescaleDB, Redis, MinIO, MLflow
	$(COMPOSE) up -d kafka zookeeper timescaledb redis minio mlflow
	@echo "Waiting for services to be healthy..."
	@sleep 10
	$(COMPOSE) ps

run-all: ## Start the complete faultscope stack
	$(COMPOSE) up -d
	@echo "✓ All services started. Dashboard: http://localhost:8501"
	@echo "  Inference API:  http://localhost:8000/docs"
	@echo "  Alert API:      http://localhost:8001/docs"
	@echo "  MLflow UI:      http://localhost:5000"
	@echo "  Grafana:        http://localhost:3000"

stop: ## Stop all services
	$(COMPOSE) down

stop-clean: ## Stop all services and remove volumes (destructive)
	$(COMPOSE) down -v

logs: ## Tail logs for all services
	$(COMPOSE) logs -f

##@ Data & Training

seed: ## Load demo data (NASA C-MAPSS + synthetic engines)
	$(UV) run python scripts/seed_demo_data.py

train: ## Train LSTM and Random Forest models
	$(UV) run python scripts/train_models.py

health: ## Check health of all running services
	$(UV) run python scripts/health_check.py

##@ Cleanup

clean: ## Remove build artifacts, caches, reports
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf reports/ dist/ .coverage
	@echo "✓ Cleaned"
