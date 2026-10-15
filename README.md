# 🧠 GraphRAG Vectorization API

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

API de ingestão, vetorização e gerenciamento de bases de conhecimento **GraphRAG** (Retrieval-Augmented Generation estruturado sobre Grafos).

O projeto é **genérico e desacoplado de provedores**: o armazenamento de objetos, o repositório do grafo, os embeddings e os resumos contextuais são selecionados por **variáveis de ambiente** (`.env`), com implementações *mock*/locais prontas para uso — nenhuma credencial ou dependência proprietária é necessária para rodar.

Backends de **nuvem já prontos e opcionais** — **AWS S3** (bucket) e **APIs compatíveis com OpenAI** (embeddings e resumos) — são ativados apenas preenchendo o `.env`. Sem isso, tudo roda **100% local, offline e sem nenhuma configuração**.

---

## 🏗️ Arquitetura

```
                     +-----------------------+
                     |   Documentos (PDFs)   |
                     +-----------+-----------+
                                 |
                                 v
                   +-------------+-------------+
                   |   FastAPI Ingestion API   |
                   |   (vetorizacao_api)       |
                   +------+--------------+-----+
                          |              |
          Fatiamento      |              | Embeddings + Resumo
          (LlamaIndex)    v              v
                   +------+--------------+-----+
                   |   camadas genericas       |
                   |  EmbeddingProvider        |  <- hash (mock) | openai (nuvem)
                   |  Summarizer               |  <- extractive | openai (nuvem)
                   +------+--------------+-----+
                          |              |
         Nos e Arestas    |              | Backup do Grafo (.json) + PDFs
                          v              v
                   +------+---------+  +----------------+
                   | GraphRepository|  |  ObjectStore   |
                   | sqlite | memory|  | local|memory|s3|
                   +----------------+  +----------------+
```

Todas as camadas são interfaces abstratas (`abc.ABC`) com fábricas em `app/storage.py` e `app/embeddings.py`. Trocar de backend = implementar a interface e registrar na fábrica.

---

## ✨ Funcionalidades

* **Ingestão inteligente com retenção de contexto:** durante o fatiamento, cada *chunk* é contextualizado frente a uma janela deslizante do documento antes de vetorizar.
* **Construção de grafos semânticos (GraphRAG):** **NetworkX** calcula relações de adjacência por similaridade de cosseno (limiar configurável, padrão `> 0.85`).
* **Persistência genérica:** upserts em lote de nós/arestas via `GraphRepository` (SQLite ou memória) e serialização do grafo via `ObjectStore` (disco ou memória).
* **Tarefas assíncronas:** processamentos longos (lote, deleção, sobrescrita) rodam em *threads* com status consultável por `task_id`.
* **Zero segredos por padrão:** nenhum `.env` real, chave ou wallet é versionado. O repositório roda out-of-the-box com os backends *mock*.
* **Nuvem em 1 minuto (opcional):** AWS S3 e APIs compatíveis com OpenAI já implementados — 1 `pip install` + preencher o `.env` (veja [Conectando sua nuvem](#️-conectando-sua-nuvem-opcional)).

---

## 📁 Estrutura de Diretórios

```
Motor_vetorizacao/
├── .gitignore                         # Ignora segredos (.env, wallet, .storage)
├── README.md
└── vetorizacao_api/                   # API FastAPI para Ingestão e Vetorização
    ├── .env.example                   # Modelo de configuração (versionado)
    ├── Dockerfile                     # Containerização da API
    ├── requirements.txt               # Dependências Python (local)
    ├── requirements-cloud.txt         # Dependências opcionais de nuvem (AWS/OpenAI)
    └── app/
        ├── __init__.py
        ├── main.py                    # Servidor FastAPI e registro de rotas
        ├── config.py                  # Carregamento do .env e Settings
        ├── storage.py                 # Interfaces + backends ObjectStore/GraphRepository
        ├── embeddings.py              # Interfaces + backends Embedding/Summarizer
        ├── documents.py                # Endpoints de upload e ingestão (individual/lote)
        ├── deletion.py                 # Endpoint para purgar documentos
        ├── replacement.py              # Endpoint para sobrescrever conteúdos
        ├── tasks.py                    # Monitoramento de status de tarefas
        └── pipeline.py                 # Pipeline GraphRagPipeline de GraphRAG
```

---

## ⚙️ Configuração (`.env`)

Copie o modelo e ajuste os valores. **Nenhuma variável é obrigatória** — os padrões já funcionam localmente:

```bash 
cd vetorizacao_api
cp .env.example .env
```

| Variável | Padrão | Descrição |
|---|---|---|
| `STORAGE_BACKEND` | `local` | ObjectStore: `local` (disco), `memory` (mock) ou `s3` (AWS) |
| `STORAGE_DIR` | `.storage` | Diretório raiz dos objetos quando `local` |
| `GRAPH_BACKEND` | `sqlite` | Backend do repositório do grafo: `sqlite` ou `memory` |
| `SQLITE_PATH` | `.storage/graph.db` | Arquivo do banco SQLite |
| `EMBEDDING_BACKEND` | `hash` | Embeddings: `hash` (mock offline) ou `openai` (API compatível) |
| `EMBEDDING_DIM` | `1024` | Dimensão dos vetores (usada apenas pelo backend `hash`) |
| `SUMMARIZER_BACKEND` | `extractive` | Resumo: `extractive` (offline) ou `openai` (LLM) |
| `SIMILARITY_THRESHOLD` | `0.85` | Limiar de cosseno para criar arestas |
| `BATCH_SIZE` | `10` | Tamanho do lote de gravação no repositório |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1024` / `200` | Fatiamento de texto |
| `DOCS_PREFIX` | `documentos/` | Prefixo (pasta) dos PDFs no ObjectStore |

**Nuvem — AWS S3** (ativos com `STORAGE_BACKEND=s3`; requerem `pip install -r requirements-cloud.txt`):

| Variável | Padrão | Descrição |
|---|---|---|
| `S3_BUCKET` | — | **Obrigatório** com `s3`: nome do bucket |
| `S3_REGION` | — | Região AWS (aceita também `AWS_REGION`) |
| `S3_ENDPOINT_URL` | — | Opcional: MinIO, Cloudflare R2 ou outro S3-compatível |
| `S3_PREFIX` | — | Opcional: subpasta dentro do bucket |
| `S3_ADDRESSING_STYLE` | — | `virtual` (AWS) ou `path` (MinIO) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | Credenciais (ou IAM Role/`~/.aws/credentials`) |

**Nuvem — API compatível com OpenAI** (ativos com backend `openai`):

| Variável | Padrão | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | — | Chave da API (dispensável em endpoints locais como Ollama) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Qualquer API compatível (OpenAI, OpenRouter, Groq, Ollama...) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo usado por `EMBEDDING_BACKEND=openai` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Modelo usado por `SUMMARIZER_BACKEND=openai` |

> Os arquivos `.env`, `config.env`, `*.pem` e `wallet/` estão no `.gitignore`. Nunca versione credenciais.

---

## 🚀 Como Executar

### Localmente

```bash
cd vetorizacao_api
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt

# Quer usar nuvem (AWS S3 / OpenAI)? Instale também os extras:
# pip install -r requirements-cloud.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentação Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

### Com Docker

```bash
cd vetorizacao_api
docker build -t graphrag-api .
# Imagem COM dependências de nuvem (AWS S3 / OpenAI):
# docker build --build-arg CLOUD=1 -t graphrag-api .

docker run -p 8080:8080 -v graphrag-data:/data/storage graphrag-api
# Com nuvem, passe o seu .env:
# docker run -p 8080:8080 --env-file .env -v graphrag-data:/data/storage graphrag-api
```

---

## ☁️ Conectando sua nuvem (opcional)

> **Sem nuvem? Não faça nada** — o projeto já roda 100% local.
> **Com nuvem? São 2 passos:** instalar os extras e preencher o `.env`.

**Passo 1 — dependências de nuvem (uma única vez):**

```bash
pip install -r requirements-cloud.txt
```

**Passo 2 — copie o modelo e preencha só o bloco que for usar:**

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

<details>
<summary><b>🟠 AWS S3</b> — PDFs e grafo direto no bucket</summary>
<br>

```env
STORAGE_BACKEND=s3
S3_BUCKET=meu-bucket-graphrag
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

* Em **EC2/ECS com IAM Role**, as chaves podem ficar em branco (o boto3 resolve).
* Também são aceitos `~/.aws/credentials` e `AWS_PROFILE`.
* **MinIO / Cloudflare R2 / S3-compatível:** adicione `S3_ENDPOINT_URL=...`
  (e `S3_ADDRESSING_STYLE=path` no MinIO).

</details>

<details>
<summary><b>🟢 Embeddings e resumos via API</b> (OpenAI ou compatível)</summary>
<br>

```env
EMBEDDING_BACKEND=openai
SUMMARIZER_BACKEND=openai
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1   # padrão
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_CHAT_MODEL=gpt-4o-mini
```

Funciona com **OpenAI, OpenRouter, Groq, Together** e qualquer API no mesmo
formato — basta trocar `OPENAI_BASE_URL`. Para **Ollama local** (gratuito):

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_EMBEDDING_MODEL=nomic-embed-text
OPENAI_CHAT_MODEL=llama3.2
```

</details>

> No boot a API loga `Backends ativos | storage=... | embeddings=...` —
> é a forma rápida de confirmar que a nuvem foi reconhecida.

---

## 📡 Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/api/v1/documentos/upload` | Salva um PDF (base64) no ObjectStore |
| `POST` | `/api/v1/documentos/sincronizar-bucket` | Varre `documentos/` e vetoriza em lote (async) |
| `GET`  | `/api/v1/documentos?collection=...` | Lista documentos integrados ao grafo da collection |
| `GET`  | `/api/v1/bucket/documentos` | Lista os PDFs físicos no ObjectStore |
| `DELETE` | `/api/v1/bucket/documentos/limpar?file_name=...` | Limpa a pasta (um arquivo ou tudo) |
| `DELETE` | `/api/v1/documentos` | Remove documento do repositório e do grafo (async) |
| `PUT`  | `/api/v1/documentos/sobrescrever` | Substitui um documento via base64 (async) |
| `POST` | `/api/v1/base/inicializar?collection=...` | Dispara processamento em lote (async) |
| `GET`  | `/api/v1/tarefas?task_id=...` | Consulta o status de uma tarefa |

---

## 🧩 Extensendo os backends

Cada camada é uma interface pequena e isolada:

* **Novo ObjectStore (GCS, Azure...)** — implemente `ObjectStore` (`get`, `put`, `delete`, `list`) em `app/storage.py` e registre em `build_object_store()` (o **S3/AWS** já vem pronto).
* **Novo GraphRepository (Postgres, Neo4j...)** — implemente `GraphRepository` (`salvar_lote`, `remover_documento`) e registre em `build_graph_repository()`.
* **Novo EmbeddingProvider (Cohere, HuggingFace, OCI...)** — implemente `EmbeddingProvider.embed()` em `app/embeddings.py` e registre em `build_embedding_provider()` (o **OpenAI-compatible** já vem pronto).
* **Novo Summarizer (outro LLM...)** — implemente `Summarizer.summarize()` e registre em `build_summarizer()` (o **OpenAI-compatible** já vem pronto).

Depois, basta apontar a variável correspondente no `.env`.

---

## 🛠️ Tecnologias

* **[FastAPI](https://fastapi.tiangolo.com/):** framework web de alto desempenho.
* **[NetworkX](https://networkx.org/):** grafos semânticos.
* **[LlamaIndex](https://www.llamaindex.ai/):** fatiamento e estruturação de documentos.
* **[scikit-learn](https://scikit-learn.org/):** similaridade de cosseno.
* **[pypdf](https://pypdf.readthedocs.io/):** extração de texto de PDFs.
* **[SQLite](https://www.sqlite.org/):** repositório do grafo embutido (stdlib, sem serviço externo).
* **[boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) *(opcional)*:** backend S3 (AWS e provedores compatíveis).
* **[requests](https://requests.readthedocs.io/) *(opcional)*:** chamadas a APIs compatíveis com OpenAI.
