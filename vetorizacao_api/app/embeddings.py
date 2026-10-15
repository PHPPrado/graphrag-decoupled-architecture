"""
Geração de embeddings e resumos contextuais (desacoplada de provedores).

Dois contratos são expostos:

1. ``EmbeddingProvider`` — converte texto em um vetor denso.
   Backends: ``hash`` (determinístico, offline, sem chamadas externas)
   ou ``openai`` (API compatível com OpenAI — requer ``requests``).

2. ``Summarizer`` — gera o resumo contextual de um chunk.
   Backends: ``extractive`` (extrativo simples, offline)
   ou ``openai`` (LLM via API compatível com OpenAI).

Todos são selecionáveis por variáveis de ambiente (``.env``) e plugáveis:
para um provedor novo (Cohere, HuggingFace, OCI, self-hosted...), basta
implementar a interface e registrá-la na fábrica ``build_*`` ao final.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MAX_TOKENS_POR_VETOR = 512


# =========================================================================
# 0. HTTP — chamadas a APIs compatíveis com OpenAI (compartilhado)
# =========================================================================
def _post_json(url: str, payload: dict, api_key: str, timeout: int = 60) -> dict:
    """
    POST JSON em uma API compatível com OpenAI.

    Requer o pacote ``requests`` (pip install -r requirements-cloud.txt).
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Backends 'openai' requerem o pacote 'requests'. "
            "Instale as dependências de nuvem com: "
            "pip install -r requirements-cloud.txt"
        ) from exc

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resposta = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Falha de rede ao chamar {url}: {exc}") from exc
    if resposta.status_code != 200:
        raise RuntimeError(
            f"API {url} respondeu HTTP {resposta.status_code}: "
            f"{resposta.text[:500]}"
        )
    return resposta.json()


# =========================================================================
# 1. EMBEDDINGS
# =========================================================================
class EmbeddingProvider(ABC):
    """Contrato mínimo de um gerador de embeddings."""

    dimension: int

    @abstractmethod
    def embed(self, text: str, max_tokens: int = _MAX_TOKENS_POR_VETOR) -> list[float] | None:
        """Retorna o vetor denso do texto ou ``None`` se o texto for vazio."""


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Mock determinístico baseado em hashing (feature hashing / "hashing trick").

    - Offline: nenhuma chamada de rede, sem chave de API.
    - Determinístico: o mesmo texto gera sempre o mesmo vetor.
    - Normalizado (L2), compatível com similaridade de cosseno.

    Textos longos são fatiados em blocos de ``max_tokens`` e o vetor final é
    a média dos blocos (mesma estratégia do pipeline original).
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def _vetor_de_tokens(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            hash_int = int.from_bytes(digest[:8], "big")
            index = hash_int % self.dimension
            sign = 1.0 if (hash_int >> 16) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, text: str, max_tokens: int = _MAX_TOKENS_POR_VETOR) -> list[float] | None:
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return None

        blocos = [
            tokens[i : i + max_tokens]
            for i in range(0, len(tokens), max_tokens)
        ]
        vetores = [self._vetor_de_tokens(bloco) for bloco in blocos]

        media = [0.0] * self.dimension
        for vetor in vetores:
            for i, valor in enumerate(vetor):
                media[i] += valor
        media = [v / len(vetores) for v in media]

        norma = math.sqrt(sum(v * v for v in media))
        if norma > 0:
            media = [v / norma for v in media]
        return media


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embeddings reais via API compatível com OpenAI.

    Usa ``POST {OPENAI_BASE_URL}/embeddings`` — funciona com a própria OpenAI,
    OpenRouter, Groq, Together, Ollama local, vLLM etc. A dimensão do vetor é
    a do modelo escolhido em ``OPENAI_EMBEDDING_MODEL`` (não usa EMBEDDING_DIM).
    """

    def __init__(
        self, api_key: str, base_url: str, model: str, dimension: int
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension

    def embed(
        self, text: str, max_tokens: int = _MAX_TOKENS_POR_VETOR
    ) -> list[float] | None:
        texto = (text or "").strip()
        if not texto:
            return None
        data = _post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": texto},
            self.api_key,
        )
        itens = data.get("data") or []
        if not itens or not itens[0].get("embedding"):
            raise RuntimeError(
                f"Resposta inesperada de {self.base_url}/embeddings: "
                f"{str(data)[:300]}"
            )
        vetor = itens[0]["embedding"]
        self.dimension = len(vetor)
        return vetor


# =========================================================================
# 2. RESUMO CONTEXTUAL
# =========================================================================
class Summarizer(ABC):
    """Contrato mínimo de um gerador de resumo contextual."""

    @abstractmethod
    def summarize(self, chunk: str, context: str = "") -> str:
        """Retorna um resumo curto (1-2 frases) do chunk, ou string vazia."""


class ExtractiveSummarizer(Summarizer):
    """
    Resumo extrativo offline: retorna a abertura do próprio chunk,
    truncada em ``max_chars``. Substitui o chat LLM no modo genérico.
    """

    def __init__(self, max_chars: int = 200) -> None:
        self.max_chars = max_chars

    def summarize(self, chunk: str, context: str = "") -> str:
        texto = " ".join((chunk or "").split())
        if not texto:
            return ""
        if len(texto) <= self.max_chars:
            return texto
        corte = texto[: self.max_chars]
        fim_frase = max(corte.rfind("."), corte.rfind("!"), corte.rfind("?"))
        if fim_frase > self.max_chars // 2:
            return corte[: fim_frase + 1]
        return corte.rstrip() + "..."


class OpenAISummarizer(Summarizer):
    """
    Resumo contextual via LLM (``POST {OPENAI_BASE_URL}/chat/completions``).

    Requer ``OPENAI_API_KEY`` (ou um endpoint local sem chave, ex.: Ollama) e
    o pacote ``requests``. O modelo é definido em ``OPENAI_CHAT_MODEL``.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def summarize(self, chunk: str, context: str = "") -> str:
        texto = " ".join((chunk or "").split())
        if not texto:
            return ""
        contexto = " ".join((context or "").split())
        mensagem_usuario = (
            f"Contexto do documento:\n{contexto}\n\nTrecho a resumir:\n{texto}"
            if contexto
            else f"Trecho a resumir:\n{texto}"
        )
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 150,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Resuma o trecho em 1 ou 2 frases objetivas, "
                            "em português. Responda apenas com o resumo."
                        ),
                    },
                    {"role": "user", "content": mensagem_usuario},
                ],
            },
            self.api_key,
        )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Resposta inesperada de {self.base_url}/chat/completions: "
                f"{str(data)[:300]}"
            )
        return ((choices[0].get("message") or {}).get("content") or "").strip()


# =========================================================================
# 3. FÁBRICAS (singletons configurados pelo .env)
# =========================================================================
@lru_cache(maxsize=1)
def build_embedding_provider() -> EmbeddingProvider:
    backend = settings.embedding_backend
    if backend == "hash":
        logger.info(
            f"EmbeddingProvider: backend 'hash' (mock offline, "
            f"dim={settings.embedding_dim})."
        )
        return HashEmbeddingProvider(dimension=settings.embedding_dim)
    if backend == "openai":
        if (
            not settings.openai_api_key
            and "api.openai.com" in settings.openai_base_url
        ):
            raise ValueError(
                "EMBEDDING_BACKEND=openai, mas OPENAI_API_KEY não está definido. "
                "Copie .env.example para .env e preencha OPENAI_API_KEY "
                "(ou use EMBEDDING_BACKEND=hash para rodar offline)."
            )
        logger.info(
            f"EmbeddingProvider: backend 'openai' "
            f"(model={settings.openai_embedding_model}, "
            f"base_url={settings.openai_base_url})."
        )
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dim,
        )
    raise ValueError(
        f"EMBEDDING_BACKEND inválido: '{backend}'. "
        "Use 'hash' (offline) ou 'openai' (API compatível)."
    )


@lru_cache(maxsize=1)
def build_summarizer() -> Summarizer:
    backend = settings.summarizer_backend
    if backend == "extractive":
        logger.info("Summarizer: backend 'extractive' (offline).")
        return ExtractiveSummarizer()
    if backend == "openai":
        if (
            not settings.openai_api_key
            and "api.openai.com" in settings.openai_base_url
        ):
            raise ValueError(
                "SUMMARIZER_BACKEND=openai, mas OPENAI_API_KEY não está "
                "definido. Copie .env.example para .env e preencha "
                "OPENAI_API_KEY (ou use SUMMARIZER_BACKEND=extractive "
                "para rodar offline)."
            )
        logger.info(
            f"Summarizer: backend 'openai' "
            f"(model={settings.openai_chat_model}, "
            f"base_url={settings.openai_base_url})."
        )
        return OpenAISummarizer(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_chat_model,
        )
    raise ValueError(
        f"SUMMARIZER_BACKEND inválido: '{backend}'. "
        "Use 'extractive' (offline) ou 'openai' (LLM via API)."
    )