"""
Configuração central da aplicação.

Todas as definições vêm de variáveis de ambiente carregadas de um arquivo
`.env` na raiz de `vetorizacao_api/` (veja `.env.example`). Nenhum segredo,
credencial ou endpoint é versionado neste repositório.

Os backends são 100% locais por padrão. Backends de nuvem opcionais
(AWS S3 e APIs compatíveis com OpenAI) são ativados apenas definindo as
variáveis correspondentes no `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Raiz do projeto (vetorizacao_api/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Carrega o `.env` padrão; por compatibilidade, aceita também `config.env`.
_DOTENV = BASE_DIR / ".env"
_CONFIG_ENV = BASE_DIR / "config.env"
if _DOTENV.is_file():
    load_dotenv(dotenv_path=_DOTENV, override=True)
elif _CONFIG_ENV.is_file():
    load_dotenv(dotenv_path=_CONFIG_ENV, override=True)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Configurações imutáveis lidas do ambiente."""

    # Armazenamento de objetos (grafos em JSON + PDFs)
    # -> "local" (disco) | "memory" (mock) | "s3" (AWS / S3-compatível)
    storage_backend: str
    storage_dir: Path

    # Nuvem — AWS S3 (usado quando storage_backend == "s3")
    s3_bucket: str
    s3_region: str
    s3_endpoint_url: str
    s3_prefix: str
    s3_addressing_style: str

    # Repositório de nós/arestas do grafo -> "sqlite" | "memory"
    graph_backend: str
    sqlite_path: Path

    # Vetorização / sumarização
    # embedding_backend  -> "hash" (mock offline) | "openai" (API compatível)
    # summarizer_backend -> "extractive" (offline) | "openai"
    embedding_backend: str
    embedding_dim: int
    summarizer_backend: str

    # Nuvem — APIs compatíveis com OpenAI (OpenAI, OpenRouter, Groq, Ollama...)
    openai_api_key: str
    openai_base_url: str
    openai_embedding_model: str
    openai_chat_model: str

    # Parâmetros do pipeline
    similarity_threshold: float
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    docs_prefix: str


def _build_settings() -> Settings:
    storage_dir = Path(_env_str("STORAGE_DIR", str(BASE_DIR / ".storage")))
    if not storage_dir.is_absolute():
        storage_dir = BASE_DIR / storage_dir

    sqlite_path = Path(_env_str("SQLITE_PATH", str(storage_dir / "graph.db")))
    if not sqlite_path.is_absolute():
        sqlite_path = BASE_DIR / sqlite_path

    return Settings(
        storage_backend=_env_str("STORAGE_BACKEND", "local").lower(),
        storage_dir=storage_dir,

        # Nuvem — AWS S3 (opcionais; usados apenas com STORAGE_BACKEND=s3)
        s3_bucket=_env_str("S3_BUCKET", ""),
        s3_region=_env_str(
            "S3_REGION", _env_str("AWS_REGION", _env_str("AWS_DEFAULT_REGION", ""))
        ),
        s3_endpoint_url=_env_str("S3_ENDPOINT_URL", "").rstrip("/"),
        s3_prefix=_env_str("S3_PREFIX", "").strip("/"),
        s3_addressing_style=_env_str("S3_ADDRESSING_STYLE", "").lower(),

        graph_backend=_env_str("GRAPH_BACKEND", "sqlite").lower(),
        sqlite_path=sqlite_path,
        embedding_backend=_env_str("EMBEDDING_BACKEND", "hash").lower(),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        summarizer_backend=_env_str("SUMMARIZER_BACKEND", "extractive").lower(),

        # Nuvem — APIs compatíveis com OpenAI (opcionais)
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
        openai_base_url=_env_str(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/"),
        openai_embedding_model=_env_str(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        openai_chat_model=_env_str("OPENAI_CHAT_MODEL", "gpt-4o-mini"),

        similarity_threshold=_env_float("SIMILARITY_THRESHOLD", 0.85),
        batch_size=max(1, _env_int("BATCH_SIZE", 10)),
        chunk_size=_env_int("CHUNK_SIZE", 1024),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        docs_prefix=_env_str("DOCS_PREFIX", "documentos/"),
    )


settings = _build_settings()