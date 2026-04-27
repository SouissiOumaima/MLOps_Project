# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
IMAGE_NAME  = oumaima_souissi_ds2_mlops
DOCKER_USER = oumaima_dockerhub
PYTHON      = python
PIP         = pip

# ══════════════════════════════════════════════════════════════
# Aide
# ══════════════════════════════════════════════════════════════
help:
	@echo "Commandes disponibles :"
	@echo "  make install        - Installe les dependances"
	@echo "  make prepare        - Prepare les donnees"
	@echo "  make train          - Entraine le modele"
	@echo "  make evaluate       - Evalue le modele"
	@echo "  make all            - Pipeline ML complet"
	@echo "  make format         - Formate le code (black)"
	@echo "  make lint           - Verifie la qualite (flake8)"
	@echo "  make security       - Analyse la securite (bandit)"
	@echo "  make test           - Lance les tests unitaires"
	@echo "  make coverage       - Tests avec rapport de couverture"
	@echo "  make ci             - Pipeline CI complet"
	@echo "  make clean          - Supprime les fichiers temporaires"
	@echo "  make run-api        - Lance l'API FastAPI"
	@echo "  make mlflow-ui      - Lance l'interface MLflow"
	@echo "  make monitoring-up  - Lance Elasticsearch + Kibana"
	@echo "  make monitoring-down- Arrete Elasticsearch + Kibana"
	@echo "  make docker-build   - Construit l'image Docker"
	@echo "  make docker-run     - Lance le conteneur Docker"
	@echo "  make docker-push    - Publie l'image sur Docker Hub"

# ══════════════════════════════════════════════════════════════
# Installation
# ══════════════════════════════════════════════════════════════
install:
	$(PIP) install -r requirements.txt

# ══════════════════════════════════════════════════════════════
# Pipeline ML
# ══════════════════════════════════════════════════════════════
prepare:
	$(PYTHON) main.py --prepare

train:
	$(PYTHON) main.py --train

evaluate:
	$(PYTHON) main.py --evaluate

all:
	$(PYTHON) main.py --all

# ══════════════════════════════════════════════════════════════
# CI — Qualité du code
# ══════════════════════════════════════════════════════════════
format:
	black src/ main.py app.py

lint:
	flake8 src/ main.py app.py --max-line-length=100

security:
	bandit -r src/ -ll

# ── Tests ─────────────────────────────────────────────────────
test:
	pytest tests/ -v

coverage:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# ── CI complète : format + lint + security + tests + coverage ──
ci: format lint security coverage
	@echo ""
	@echo "======================================"
	@echo "  CI complete — tout est bon !"
	@echo "======================================"

# ══════════════════════════════════════════════════════════════
# Nettoyage
# ══════════════════════════════════════════════════════════════
clean:
	@echo "Nettoyage des fichiers temporaires..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"       -delete 2>/dev/null || true
	find . -type f -name "*.pyo"       -delete 2>/dev/null || true
	find . -type f -name ".coverage"   -delete 2>/dev/null || true
	find . -type d -name "htmlcov"     -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "mlruns"      -exec rm -rf {} + 2>/dev/null || true
	@echo "Nettoyage termine !"

# ══════════════════════════════════════════════════════════════
# API FastAPI
# ══════════════════════════════════════════════════════════════
run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# ══════════════════════════════════════════════════════════════
# MLflow
# ══════════════════════════════════════════════════════════════
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

mlflow-server:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 --port 5000

# ══════════════════════════════════════════════════════════════
# Monitoring — Elasticsearch + Kibana
# ══════════════════════════════════════════════════════════════
monitoring-up:
	docker-compose up -d

monitoring-down:
	docker-compose down

# ══════════════════════════════════════════════════════════════
# Docker
# ══════════════════════════════════════════════════════════════
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --env-file .env -p $(API_PORT):$(API_PORT) $(IMAGE_NAME)

docker-push:
	docker tag $(IMAGE_NAME) $(DOCKER_USER)/$(IMAGE_NAME):latest
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest

# ── Déclarer les cibles non-fichiers ──────────────────────────
.PHONY: help install prepare train evaluate all \
        format lint security test coverage ci clean \
        run-api mlflow-ui mlflow-server \
        monitoring-up monitoring-down \
        docker-build docker-run docker-push
compare:
	python main.py --compare