"""
Camada de persistência genérica (desacoplada de qualquer provedor).

Dois contratos são expostos:

1. ``ObjectStore``  — armazenamento de objetos (JSON do grafo + PDFs).
   Backends: ``local`` (arquivos em disco), ``memory`` (mock em RAM)
   ou ``s3`` (bucket AWS S3 / compatível — requer ``boto3``).

2. ``GraphRepository`` — repositório de nós/arestas do grafo.
   Backends: ``sqlite`` (arquivo leve, stdlib) ou ``memory`` (mock em RAM).

O backend é escolhido exclusivamente por variáveis de ambiente (``.env``).
Para usar outro provedor (GCS, Azure, Postgres...), basta implementar a
interface correspondente e registrá-la nas fábricas ``build_*`` ao final.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


# =========================================================================
# 1. ARMAZENAMENTO DE OBJETOS (bucket / grafo / PDFs)
# =========================================================================
class ObjectStore(ABC):
    """Contrato mínimo de um armazenamento de objetos (blob storage)."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Retorna os bytes do objeto ou ``None`` se não existir."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Grava (create/replace) o objeto informado."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove o objeto; silencioso se não existir."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Lista as chaves que iniciam com ``prefix``, em ordem crescente."""


class MemoryObjectStore(ObjectStore):
    """Mock em memória — ideal para testes unitários e ambientes efêmeros."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._objects.get(key)

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            self._objects[key] = data

    def delete(self, key: str) -> None:
        with self._lock:
            self._objects.pop(key, None)

    def list(self, prefix: str = "") -> list[str]:
        with self._lock:
            return sorted(k for k in self._objects if k.startswith(prefix))


class LocalObjectStore(ObjectStore):
    """Persiste os objetos em disco — substitui um Object Storage qualquer."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _resolve(self, key: str) -> Path:
        clean = key.replace("\\", "/").strip("/")
        if not clean:
            raise ValueError("Chave de objeto vazia.")
        path = (self.root / clean).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Chave de objeto inválida (fora da raiz): {key}")
        return path

    def get(self, key: str) -> bytes | None:
        path = self._resolve(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def put(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        with self._lock:
            if path.is_file():
                path.unlink()

    def list(self, prefix: str = "") -> list[str]:
        results: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                results.append(key)
        return sorted(results)


class S3ObjectStore(ObjectStore):
    """
    Bucket S3 (AWS ou qualquer provedor compatível via ``S3_ENDPOINT_URL``).

    - Requer o pacote ``boto3``: ``pip install -r requirements-cloud.txt``
    - Credenciais: ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` no ``.env``,
      em ``~/.aws/credentials`` ou IAM Role (EC2/ECS/Cloud9) — o boto3 resolve.
    - ``S3_PREFIX`` (opcional) grava tudo dentro de uma subpasta do bucket.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "",
        endpoint_url: str = "",
        prefix: str = "",
        addressing_style: str = "",
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "STORAGE_BACKEND=s3 requer o pacote 'boto3'. "
                "Instale as dependências de nuvem com: "
                "pip install -r requirements-cloud.txt"
            ) from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        config = None
        if addressing_style in ("path", "virtual"):
            config = Config(s3={"addressing_style": addressing_style})
        self._client = boto3.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint_url or None,
            config=config,
        )

    # --- helpers de chave -------------------------------------------------
    def _full_key(self, key: str) -> str:
        clean = (key or "").strip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def _strip_prefix(self, full_key: str) -> str:
        if self.prefix and full_key.startswith(f"{self.prefix}/"):
            return full_key[len(self.prefix) + 1 :]
        return full_key

    @staticmethod
    def _error_code(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            return str((response.get("Error") or {}).get("Code", ""))
        return ""

    # --- contrato ObjectStore ---------------------------------------------
    def get(self, key: str) -> bytes | None:
        try:
            resp = self._client.get_object(
                Bucket=self.bucket, Key=self._full_key(key)
            )
            return resp["Body"].read()
        except Exception as exc:
            if self._error_code(exc) in ("NoSuchKey", "NotFound", "404"):
                return None
            raise

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def delete(self, key: str) -> None:
        # No S3, delete de objeto inexistente é silencioso por natureza.
        self._client.delete_object(Bucket=self.bucket, Key=self._full_key(key))

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = "/".join(
            parte for parte in (self.prefix, (prefix or "").strip("/")) if parte
        )
        names: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                names.append(self._strip_prefix(obj["Key"]))
        return sorted(names)


# =========================================================================
# 2. REPOSITÓRIO DO GRAFO (nós e arestas)
# =========================================================================
class GraphRepository(ABC):
    """Contrato mínimo de persistência do grafo (vértices + arestas)."""

    @abstractmethod
    def salvar_lote(self, nodes: list[dict], edges: list[dict], collection: str) -> None:
        """
        Upsert em lote.
        - ``nodes``: [{"id": str, "props": dict}, ...]
        - ``edges``: [{"source": str, "target": str, "relation": str, "weight": float}, ...]
        """

    @abstractmethod
    def remover_documento(self, doc_id: str, collection: str) -> tuple[int, int]:
        """
        Remove os nós do documento (props["doc_id"] == doc_id) dentro da collection
        informada, junto com as arestas que tocam esses nós.
        Retorna ``(nos_removidos, arestas_removidas)``.
        """


class MemoryGraphRepository(GraphRepository):
    """Mock em memória — descartado a cada reinício do processo."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, dict]] = {}
        self._edges: dict[str, dict[tuple[str, str, str], dict]] = {}
        self._lock = threading.Lock()

    def salvar_lote(self, nodes: list[dict], edges: list[dict], collection: str) -> None:
        with self._lock:
            bucket_nodes = self._nodes.setdefault(collection, {})
            bucket_edges = self._edges.setdefault(collection, {})
            for node in nodes:
                bucket_nodes[str(node["id"])] = dict(node)
            for edge in edges:
                key = (str(edge["source"]), str(edge["target"]), str(edge.get("relation", "")))
                bucket_edges[key] = dict(edge)

    def remover_documento(self, doc_id: str, collection: str) -> tuple[int, int]:
        with self._lock:
            bucket_nodes = self._nodes.get(collection, {})
            bucket_edges = self._edges.get(collection, {})

            alvos = {
                node_id
                for node_id, node in bucket_nodes.items()
                if (node.get("props") or {}).get("doc_id") == doc_id
            }
            arestas_removidas = sum(
                1
                for key in list(bucket_edges)
                if key[0] in alvos or key[1] in alvos
            )
            for key in [k for k in bucket_edges if k[0] in alvos or k[1] in alvos]:
                del bucket_edges[key]
            for node_id in alvos:
                bucket_nodes.pop(node_id, None)

            return len(alvos), arestas_removidas


class SQLiteGraphRepository(GraphRepository):
    """Persistência local em SQLite (stdlib) — substitui o banco vetorial."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS vertices_docs (
                    node_id     TEXT NOT NULL,
                    collection_acesso TEXT NOT NULL,
                    properties  TEXT NOT NULL,
                    PRIMARY KEY (collection_acesso, node_id)
                );
                CREATE TABLE IF NOT EXISTS arestas_links (
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    tipo        TEXT NOT NULL DEFAULT 'semantically_related',
                    label       TEXT NOT NULL DEFAULT '0',
                    collection_acesso TEXT NOT NULL,
                    PRIMARY KEY (collection_acesso, source_id, target_id, tipo)
                );
                CREATE INDEX IF NOT EXISTS idx_vertices_collection ON vertices_docs (collection_acesso);
                CREATE INDEX IF NOT EXISTS idx_arestas_collection ON arestas_links (collection_acesso);
                """
            )

    def salvar_lote(self, nodes: list[dict], edges: list[dict], collection: str) -> None:
        node_rows = [
            (
                str(node["id"]),
                collection,
                json.dumps(node.get("props") or {}, ensure_ascii=False),
            )
            for node in nodes
        ]
        edge_rows = [
            (
                str(edge["source"]),
                str(edge["target"]),
                str(edge.get("relation", "semantically_related")),
                str(edge.get("weight", 0)),
                collection,
            )
            for edge in edges
        ]

        with self._lock, self._connect() as conn:
            if node_rows:
                conn.executemany(
                    """
                    INSERT INTO vertices_docs (node_id, collection_acesso, properties)
                    VALUES (?, ?, ?)
                    ON CONFLICT (collection_acesso, node_id)
                    DO UPDATE SET properties = excluded.properties
                    """,
                    node_rows,
                )
            if edge_rows:
                conn.executemany(
                    """
                    INSERT INTO arestas_links
                        (source_id, target_id, tipo, label, collection_acesso)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (collection_acesso, source_id, target_id, tipo)
                    DO UPDATE SET label = excluded.label
                    """,
                    edge_rows,
                )
            conn.commit()

    def remover_documento(self, doc_id: str, collection: str) -> tuple[int, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, properties FROM vertices_docs WHERE collection_acesso = ?",
                (collection,),
            ).fetchall()

            alvos: set[str] = set()
            for row in rows:
                try:
                    props = json.loads(row["properties"])
                except (TypeError, ValueError):
                    props = {}
                if props.get("doc_id") == doc_id:
                    alvos.add(row["node_id"])

            arestas_removidas = 0
            nos_removidos = 0
            if alvos:
                placeholders = ",".join("?" for _ in alvos)
                params = list(alvos)
                cur = conn.execute(
                    f"""
                    DELETE FROM arestas_links
                    WHERE collection_acesso = ?
                      AND (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                    """,
                    [collection, *params, *params],
                )
                arestas_removidas = cur.rowcount

                cur = conn.execute(
                    f"""
                    DELETE FROM vertices_docs
                    WHERE collection_acesso = ? AND node_id IN ({placeholders})
                    """,
                    [collection, *params],
                )
                nos_removidos = cur.rowcount
                conn.commit()

            return nos_removidos, arestas_removidas


# =========================================================================
# 3. FÁBRICAS (singletons configurados pelo .env)
# =========================================================================
@lru_cache(maxsize=1)
def build_object_store() -> ObjectStore:
    backend = settings.storage_backend
    if backend == "memory":
        logger.info("ObjectStore: backend 'memory' (mock em RAM).")
        return MemoryObjectStore()
    if backend == "local":
        logger.info(f"ObjectStore: backend 'local' em {settings.storage_dir}")
        return LocalObjectStore(settings.storage_dir)
    if backend == "s3":
        if not settings.s3_bucket:
            raise ValueError(
                "STORAGE_BACKEND=s3, mas S3_BUCKET não está definido. "
                "Copie .env.example para .env e preencha S3_BUCKET "
                "(e as credenciais AWS)."
            )
        logger.info(
            f"ObjectStore: backend 's3' no bucket '{settings.s3_bucket}' "
            f"(região: {settings.s3_region or 'padrão AWS'}, "
            f"endpoint: {settings.s3_endpoint_url or 'AWS oficial'})"
        )
        return S3ObjectStore(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            prefix=settings.s3_prefix,
            addressing_style=settings.s3_addressing_style,
        )
    raise ValueError(
        f"STORAGE_BACKEND inválido: '{backend}'. Use 'local', 'memory' ou 's3'."
    )


@lru_cache(maxsize=1)
def build_graph_repository() -> GraphRepository:
    backend = settings.graph_backend
    if backend == "memory":
        logger.info("GraphRepository: backend 'memory' (mock em RAM).")
        return MemoryGraphRepository()
    if backend == "sqlite":
        logger.info(f"GraphRepository: backend 'sqlite' em {settings.sqlite_path}")
        return SQLiteGraphRepository(settings.sqlite_path)
    raise ValueError(
        f"GRAPH_BACKEND inválido: '{backend}'. Use 'sqlite' ou 'memory'."
    )