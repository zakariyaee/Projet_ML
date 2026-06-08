FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python en premier (cache Docker)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copier le code applicatif et les modèles
COPY app/ app/
COPY models/ models/
COPY src/ src/

# Port par défaut pour l'API
EXPOSE 8000

# Commande de démarrage par défaut (API)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]