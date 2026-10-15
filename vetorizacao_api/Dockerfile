# Imagem base leve
FROM python:3.11-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia os requirements e instala as dependências
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Dependências de nuvem (opcionais) — habilite com: --build-arg CLOUD=1
# (instala requirements-cloud.txt: boto3 p/ AWS S3 + requests p/ OpenAI)
ARG CLOUD=0
RUN if [ "$CLOUD" = "1" ]; then \
        pip install --no-cache-dir -r requirements-cloud.txt; \
    fi

# Copia o código fonte da aplicação
COPY app/ ./app/

# Diretório dos backends locais (ObjectStore em disco + SQLite)
ENV STORAGE_DIR=/data/storage
ENV SQLITE_PATH=/data/storage/graph.db
RUN mkdir -p /data/storage
VOLUME ["/data/storage"]

# Expoe a porta correta do FastAPI para o Gateway
EXPOSE 8080

# Comando para rodar na porta 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]