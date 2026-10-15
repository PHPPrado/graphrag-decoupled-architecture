import uuid
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.pipeline import GraphRagPipeline
from app.tasks import TAREFAS_STATUS

router = APIRouter(tags=["Gerenciamento de Documentos"])

class DeletarDocumentoPayload(BaseModel):
    doc_id: str
    collection: str

def executar_delecao_background(task_id: str, payload: DeletarDocumentoPayload):
    """
    Executa em background seguindo o padrão ouro do sobrescrever:
    1. Descobre o nome real no grafo.
    2. Apaga do repositório do grafo.
    3. Remove os nós e reconstrói as arestas do JSON em memória.
    4. Salva no ObjectStore.
    """
    try:
        TAREFAS_STATUS[task_id] = {
            "status": "processando",
            "mensagem": f"Remoção do documento '{payload.doc_id}' em curso...",
            "collection": payload.collection
        }
        
        pipeline = GraphRagPipeline(collection_destino=payload.collection)
        
        # --- BLINDAGEM DO DOC_ID (Com e sem prefixo 'documentos/') ---
        doc_alvo = payload.doc_id.strip()
        doc_alvo_com_prefixo = doc_alvo if doc_alvo.startswith("documentos/") else f"documentos/{doc_alvo}"
        doc_alvo_sem_prefixo = doc_alvo.replace("documentos/", "")
        
        # --- PASSO 1: IDENTIFICAÇÃO DO ALVO ---
        TAREFAS_STATUS[task_id]["mensagem"] = "Carregando o histórico do grafo para localizar os nós pertencentes ao documento..."
        # 1. Recupera o histórico do grafo PRIMEIRO
        historico_atual = pipeline.recuperar_historico_grafo()

        # 2. Identifica os nós do documento alvo e pesca o NOME EXATO gravado
        nodes_alvo = [
            n for n in historico_atual 
            if n.get("doc_id") in [doc_alvo, doc_alvo_com_prefixo, doc_alvo_sem_prefixo]
        ]
        
        # 3. Pega a identidade real do arquivo
        doc_id_alvo_final = nodes_alvo[0].get("doc_id") if nodes_alvo else doc_alvo

        TAREFAS_STATUS[task_id]["mensagem"] = f"Limpando os registros de '{doc_id_alvo_final}' armazenados no repositório..."


        # 4. AGORA SIM! Manda o repositório apagar usando o nome real
        pipeline.remover_documento_do_banco(doc_id=doc_id_alvo_final)

        # --- PASSO 3: LIMPEZA E RECONSTRUÇÃO DO GRAFO (EM MEMÓRIA) ---
        TAREFAS_STATUS[task_id]["mensagem"] = "Purgando nós obsoletos da estrutura em memória e gravando a versão atualizada no ObjectStore..."
        
        # 5. Limpa do JSON em memória
        nodes_restantes = [
            n for n in historico_atual 
            if n.get("doc_id") != doc_id_alvo_final
        ]

                
        ids_restantes = {n["id"] for n in nodes_restantes}
        for node in nodes_restantes:
            node["edges"] = [
                e for e in node.get("edges", []) 
                if e.get("target") in ids_restantes and e.get("source") in ids_restantes
            ]
            
        # Reconstrói e salva
        grafo_url = pipeline.reconstruir_e_salvar_grafo(nodes_restantes)
        
        TAREFAS_STATUS[task_id] = {
            "status": "concluido",
            "mensagem": f"'{payload.doc_id}' excluído; o grafo foi reconstruído e salvo.",
            "total_nos_restantes": len(nodes_restantes),
            "grafo_atualizado": grafo_url,
            "collection_afetada": payload.collection
        }
        
    except Exception as e:
        TAREFAS_STATUS[task_id] = {
            "status": "erro",
            "mensagem": f"Exclusão abortada por erro: {str(e)}",
            "collection": payload.collection
        }

@router.delete("/api/v1/documentos")
async def endpoint_deletar_documento(payload: DeletarDocumentoPayload):
    """
    Endpoint DELETE para remover um documento da base do banco e do bucket.
    Retorna um task_id para acompanhamento assíncrono.
    """
    if not payload.doc_id or not payload.collection:
        raise HTTPException(status_code=400, detail="Preencha 'doc_id' e 'collection' para prosseguir.")
    
    task_id = str(uuid.uuid4())[:8]
    
    TAREFAS_STATUS[task_id] = {
        "status": "pendente",
        "mensagem": "Remoção agendada; aguardando processamento...",
        "collection": payload.collection
    }
    
    thread = threading.Thread(target=executar_delecao_background, args=(task_id, payload))
    thread.start()
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "aceito",
            "mensagem": "Remoção delegada a uma thread em segundo plano.",
            "task_id": task_id
        }
    )