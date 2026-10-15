from fastapi import FastAPI
import uvicorn

# 1. Carrega o ambiente (.env) e as configurações centrais
from app.config import settings  # noqa: F401  (dispara o carregamento do .env)

# 1b. Log dos backends ativos (facilita saber se a nuvem está ligada)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("backends").info(
    "Backends ativos | storage=%s | grafo=%s | embeddings=%s | resumo=%s",
    settings.storage_backend,
    settings.graph_backend,
    settings.embedding_backend,
    settings.summarizer_backend,
)

# 2. Inicializa o FastAPI central
app = FastAPI(
    title="GraphRAG Vectorization API",
    description=(
        "API de ingestão, vetorização e gerenciamento da base de "
        "conhecimento GraphRAG (backends de storage/banco genéricos)."
    ),
    version="1.0.0"
)

# 3. Registra todos os módulos de rotas (Documents, Deletion, Replacement e Tasks)
from app.documents import router as documents_router
from app.deletion import router as deletion_router
from app.replacement import router as replacement_router
from app.tasks import router as tasks_router

app.include_router(documents_router)
app.include_router(deletion_router)
app.include_router(replacement_router)
app.include_router(tasks_router)

# =========================================================================
# EXECUÇÃO (Para testes via IDE/Terminal)
# =========================================================================
if __name__ == "__main__":
    print("🚀 Iniciando servidor FastAPI...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
