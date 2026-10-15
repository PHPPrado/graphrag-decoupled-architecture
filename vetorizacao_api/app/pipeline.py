"""
Pipeline de vetorização e GraphRAG (genérico, sem dependências de provedores).

Toda a persistência passa pelas interfaces de ``app.storage``
(ObjectStore + GraphRepository) e a geração de vetores/resumos pelas
interfaces de ``app.embeddings`` (EmbeddingProvider + Summarizer).
Os backends são escolhidos exclusivamente por variáveis de ambiente (`.env`).
"""

import io
import json
import logging
import sys
import hashlib

import networkx as nx
from networkx.readwrite import json_graph
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from app.config import settings
from app.embeddings import build_embedding_provider, build_summarizer
from app.storage import build_graph_repository, build_object_store

# Exibe todos os logs no console em tempo real
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)


class GraphRagPipeline:
    def __init__(self, collection_destino: str):
        self.collection_destino = collection_destino

        # 1. Backends configurados pelo .env (singletons compartilhados)
        self.object_store = build_object_store()
        self.graph_repository = build_graph_repository()
        self.embedding_provider = build_embedding_provider()
        self.summarizer = build_summarizer()

        # 2. Parâmetros do pipeline
        self.similarity_threshold = settings.similarity_threshold
        self.tamanho_batch_upload = settings.batch_size
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.prefixo_documentos = settings.docs_prefix
        self.arquivo_final_grafo = f"GRAFO_FINAL_{self.collection_destino}.json"

    # =========================================================================
    # NÚCLEO (embedding, resumo e persistência)
    # =========================================================================
    def get_mean_embedding(self, text: str, max_tokens: int = 512):
        """Gera o vetor denso do texto via EmbeddingProvider configurado."""
        try:
            return self.embedding_provider.embed(text, max_tokens=max_tokens)
        except Exception as e:
            logging.error(f"Erro ao gerar embedding: {e}")
            return None

    def gerar_resumo_contextual(self, chunk_text: str, contexto_janela: str) -> str:
        """Resumo contextual do chunk via Summarizer configurado."""
        try:
            return self.summarizer.summarize(chunk_text, contexto_janela)
        except Exception as e:
            logging.error(f"Erro ao gerar resumo contextual: {e}")
            return ""

    def salvar_lote_grafo(self, lista_nodes: list):
        """Faz o upsert de nós e arestas no GraphRepository configurado."""
        if not lista_nodes:
            return

        nodes = []
        edges = []
        for item in lista_nodes:
            if not item.get("embedding"):
                continue
            nodes.append(
                {
                    "id": str(item["id"]),
                    "props": {k: v for k, v in item.items() if k != "edges"},
                }
            )
            for edge in item.get("edges", []):
                edges.append(
                    {
                        "source": str(edge["source"]),
                        "target": str(edge["target"]),
                        "relation": str(edge.get("relation", "semantically_related")),
                        "weight": float(edge.get("weight", 0.0)),
                    }
                )

        self.graph_repository.salvar_lote(
            nodes=nodes, edges=edges, collection=self.collection_destino
        )

    # =========================================================================
    # HISTÓRICO DO GRAFO (ObjectStore)
    # =========================================================================
    def recuperar_historico_grafo(self) -> list:
        """
        Busca o JSON do grafo histórico no ObjectStore configurado.
        Se o arquivo não existir ou a base estiver vazia, retorna lista vazia.
        """
        object_name = f"{self.collection_destino}/{self.arquivo_final_grafo}"
        logging.info(
            f"🧠 Buscando memória existente para a collection: {self.collection_destino} "
            f"(arquivo: {object_name})..."
        )
        try:
            raw = self.object_store.get(object_name)
            if raw is None:
                logging.warning(
                    "⚠️ Aviso ao recuperar histórico (a base pode estar vazia): "
                    "arquivo não encontrado."
                )
                return []

            conteudo_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            historico = json.loads(conteudo_str)

            # O NetworkX serializa os nós na chave "nodes"
            if isinstance(historico, dict):
                historico = historico.get("nodes", historico.get("nos", []))

            logging.info(
                f"✅ Memória carregada: {len(historico)} nós recuperados do storage."
            )
            return historico
        except Exception as e:
            logging.warning(
                f"⚠️ Aviso ao recuperar histórico (a base pode estar vazia ou o "
                f"arquivo não existir): {str(e)}"
            )
            return []

    def reconstruir_e_salvar_grafo(self, todos_nodes: list) -> str:
        """Serializa o grafo NetworkX completo e grava no ObjectStore."""
        G_final = nx.MultiDiGraph()
        for item in todos_nodes:
            G_final.add_node(
                item["id"],
                text=item["text"],
                doc_id=item["doc_id"],
                embedding=item["embedding"],
            )
            for edge in item.get("edges", []):
                G_final.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge["weight"],
                    relation=edge["relation"],
                )

        final_json_data = json_graph.node_link_data(G_final)
        json_bytes = json.dumps(final_json_data, ensure_ascii=False).encode("utf-8")

        object_name = f"{self.collection_destino}/{self.arquivo_final_grafo}"
        self.object_store.put(object_name, json_bytes)
        return object_name

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================
    def processar_novo_documento(
        self,
        pdf_bytes: bytes,
        file_name: str,
        historico_grafo: list,
        status_callback=None,
    ):
        """
        Recebe um documento avulso via upload, fatia, contextualiza,
        vetoriza, conecta por similaridade (GraphRAG) e persiste no repositório.
        """
        logging.info(f"🔍 [DEBUG] Iniciando processamento do documento: {file_name}")
        logging.info(f"🔍 [DEBUG] Tamanho dos bytes recebidos: {len(pdf_bytes)} bytes")

        reader = PdfReader(io.BytesIO(pdf_bytes))
        texto_extraido = "".join(
            [page.extract_text() + "\n" for page in reader.pages]
        )

        doc = Document(text=texto_extraido, metadata={"file_name": file_name})
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        raw_nodes = node_parser.get_nodes_from_documents([doc], show_progress=False)

        total_chunks = len(raw_nodes)
        logging.info(
            f"📄 PDF fatiado com sucesso! Total de chunks gerados para processar: "
            f"{total_chunks}"
        )

        vetor_antigos = (
            [n["embedding"] for n in historico_grafo] if historico_grafo else []
        )
        ids_antigos = [n["id"] for n in historico_grafo] if historico_grafo else []

        buffer_upload = []
        novos_processados = []

        for i, node in enumerate(raw_nodes):
            logging.info(f"⚙️ Processando chunk [{i + 1}/{total_chunks}]...")
            if status_callback:
                pct = (
                    round(((i + 1) / total_chunks) * 100, 1)
                    if total_chunks > 0
                    else 0
                )
                status_callback(
                    f"Vetorizando '{file_name}': trecho {i + 1}/{total_chunks} ({pct}% concluído)"
                )

            assinatura = f"{file_name}_{node.text[:50]}"
            node.id_ = f"node_{hashlib.sha256(assinatura.encode()).hexdigest()}"

            # Contexto (janela deslizante)
            start, end = max(0, i - 3), min(total_chunks, i + 4)
            contexto_janela = "\n...\n".join([n.text for n in raw_nodes[start:end]])

            resumo = self.gerar_resumo_contextual(node.text, contexto_janela)
            texto_final = (
                f"{resumo}\n\nConteúdo Original:\n{node.text}" if resumo else node.text
            )

            # Embedding
            embedding_vector = self.get_mean_embedding(texto_final)
            if not embedding_vector:
                logging.warning(
                    f"⚠️ Chunk {i + 1} ignorado (falha ao gerar embedding)."
                )
                continue

            # Arestas por similaridade de cosseno (GraphRAG)
            arestas = []
            if vetor_antigos:
                sims = cosine_similarity([embedding_vector], vetor_antigos)[0]
                for idx_ant, score in enumerate(sims):
                    if score > self.similarity_threshold:
                        arestas.append(
                            {
                                "source": node.id_,
                                "target": ids_antigos[idx_ant],
                                "weight": float(score),
                                "relation": "semantically_related",
                            }
                        )

            dado_completo = {
                "id": node.id_,
                "text": texto_final,
                "doc_id": file_name,
                "embedding": list(embedding_vector),
                "edges": arestas,
            }

            # Atualiza estado para o próximo nó
            vetor_antigos.append(embedding_vector)
            ids_antigos.append(node.id_)
            novos_processados.append(dado_completo)
            buffer_upload.append(dado_completo)

            if len(buffer_upload) >= self.tamanho_batch_upload:
                logging.info("💾 Enviando batch parcial para o repositório do grafo...")
                self.salvar_lote_grafo(buffer_upload)
                buffer_upload = []

        if buffer_upload:
            logging.info("💾 Enviando lote final para o repositório do grafo...")
            self.salvar_lote_grafo(buffer_upload)

        logging.info(
            f"✅ Processamento do documento '{file_name}' concluído com sucesso!"
        )
        return novos_processados

    def processar_lote_do_bucket(self, status_callback=None):
        """
        Varre o prefixo 'documentos/' no ObjectStore, baixa os PDFs,
        vetoriza tudo em lote e reconstrói o grafo unificado.
        """
        prefixo_pasta = self.prefixo_documentos
        logging.info(
            f"📂 Listando PDFs na pasta '{prefixo_pasta}' do ObjectStore..."
        )

        nomes = [
            nome
            for nome in self.object_store.list(prefixo_pasta)
            if nome.lower().endswith(".pdf")
        ]
        total_arquivos = len(nomes)

        total_processados = 0
        total_nos = 0

        historico_atual = self.recuperar_historico_grafo()

        for idx_arq, nome in enumerate(nomes):
            logging.info(f"📥 Baixando {nome} do ObjectStore...")
            if status_callback:
                status_callback(
                    f"Documento {idx_arq + 1}/{total_arquivos} - transferindo '{nome}' do ObjectStore..."
                )

            pdf_bytes = self.object_store.get(nome)
            if pdf_bytes is None:
                logging.warning(f"⚠️ Arquivo '{nome}' não encontrado, ignorando.")
                continue

            def doc_callback(mensagem_doc, _idx=idx_arq):
                if status_callback:
                    status_callback(
                        f"Documento {_idx + 1}/{total_arquivos} - {mensagem_doc}"
                    )

            novos_nodes = self.processar_novo_documento(
                pdf_bytes=pdf_bytes,
                file_name=nome,
                historico_grafo=historico_atual,
                status_callback=doc_callback,
            )

            historico_atual.extend(novos_nodes)
            total_processados += 1
            total_nos += len(novos_nodes)

        if total_processados > 0:
            logging.info(
                "🏁 Lote finalizado! Salvando o grafo completo atualizado..."
            )
            self.reconstruir_e_salvar_grafo(historico_atual)

        return {
            "total_docs_processados": total_processados,
            "total_nos": total_nos,
        }

    # =========================================================================
    # EXCLUSÃO DE DOCUMENTOS (GraphRepository)
    # =========================================================================
    def remover_documento_do_banco(self, doc_id: str) -> tuple[int, int]:
        """
        Remove os nós do documento e as arestas que tocam esses nós,
        restrito à collection/tenant atual, via GraphRepository configurado.
        """
        logging.info(
            f"🗑️ Iniciando exclusão do documento '{doc_id}' para a collection "
            f"'{self.collection_destino}'..."
        )
        doc_str = doc_id.strip()
        try:
            nos_removidos, arestas_removidas = self.graph_repository.remover_documento(
                doc_id=doc_str, collection=self.collection_destino
            )
            logging.info(
                f"✅ Exclusão concluída para a collection '{self.collection_destino}': "
                f"{nos_removidos} nós e {arestas_removidas} arestas removidos."
            )
            return nos_removidos, arestas_removidas
        except Exception as e:
            logging.error(f"❌ Erro ao deletar do repositório: {e}")
            raise e

    # =========================================================================
    # ARQUIVOS PDF NO ObjectStore
    # =========================================================================
    def fazer_upload_pdf_bucket(self, pdf_bytes: bytes, file_name: str) -> str:
        """
        Faz o upload físico de um arquivo PDF para a pasta 'documentos/'.
        Garante que o nome tenha o prefixo correto.
        """
        doc_puro = file_name.replace(self.prefixo_documentos, "").strip()
        object_name = f"{self.prefixo_documentos}{doc_puro}"
        logging.info(f"📤 Subindo arquivo físico para o ObjectStore: {object_name}...")
        self.object_store.put(object_name, pdf_bytes)
        return object_name

    def remover_pdf_do_bucket(self, file_name: str):
        """Deleta o arquivo PDF correspondente da pasta 'documentos/'."""
        doc_puro = file_name.replace(self.prefixo_documentos, "").strip()
        object_name = f"{self.prefixo_documentos}{doc_puro}"
        logging.info(f"🗑️ Deletando arquivo físico do ObjectStore: {object_name}...")
        self.object_store.delete(object_name)
        logging.info(f"✅ Arquivo {object_name} excluído com sucesso.")

    def listar_arquivos_pdf_bucket(self) -> list:
        """Lista todos os PDFs fisicamente na pasta 'documentos/'."""
        prefixo_pasta = self.prefixo_documentos
        logging.info(f"📂 Listando arquivos na pasta '{prefixo_pasta}'...")
        try:
            nomes = [
                nome
                for nome in self.object_store.list(prefixo_pasta)
                if nome.lower().endswith(".pdf")
            ]
            return nomes
        except Exception as e:
            logging.error(f"Erro ao listar PDFs: {e}")
            return []

    def limpar_pasta_pdf_bucket(self, file_name: str = None) -> int:
        """
        Limpa a pasta 'documentos/'.
        Se ``file_name`` for informado, remove apenas esse arquivo.
        Caso contrário, remove TODOS os PDFs da pasta.
        Retorna o total de arquivos excluídos.
        """
        if file_name:
            self.remover_pdf_do_bucket(file_name)
            return 1

        prefixo_pasta = self.prefixo_documentos
        logging.info(f"🗑️ Limpando TODOS os arquivos da pasta '{prefixo_pasta}'...")
        try:
            objetos = [
                nome
                for nome in self.object_store.list(prefixo_pasta)
                if nome.lower().endswith(".pdf")
            ]
            excluidos = 0
            for obj_name in objetos:
                try:
                    self.object_store.delete(obj_name)
                    excluidos += 1
                except Exception as e:
                    logging.warning(f"Falha ao deletar {obj_name}: {e}")
            return excluidos
        except Exception as e:
            logging.error(f"Erro ao varrer e limpar a pasta: {e}")
            return 0