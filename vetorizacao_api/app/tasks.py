import uuid
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.pipeline import GraphRagPipeline

router = APIRouter(tags=["Tarefas em Segundo Plano"])

# Dicionário em memória para rastrear o progresso das tarefas
# Exemplo: { "a1b2c3d4": { "status": "processando", "mensagem": "...", "collection": "..." } }
TAREFAS_STATUS = {}

def executar_lote_background(task_id: str, collection: str):
    """
    Função executada em thread paralela para não travar a API principal.
    """
    try:
        TAREFAS_STATUS[task_id] = {
            "status": "processando",
            "mensagem": "Abrindo conexão com o ObjectStore e preparando o processamento em lote...",
            "collection": collection
        }
        
        pipeline = GraphRagPipeline(collection_destino=collection)
        
        def status_callback(msg: str):
            TAREFAS_STATUS[task_id]["mensagem"] = msg
            
        resultado = pipeline.processar_lote_do_bucket(status_callback=status_callback)
        
        TAREFAS_STATUS[task_id] = {
            "status": "concluido",
            "mensagem": "Lote processado até o fim, sem falhas.",
            "total_documentos": resultado.get("total_docs_processados", 0),
            "total_nos_gerados": resultado.get("total_nos", 0),
            "collection_destino": collection
        }
    except Exception as e:
        TAREFAS_STATUS[task_id] = {
            "status": "erro",
            "mensagem": f"O lote foi interrompido por um erro: {str(e)}",
            "collection": collection
        }

@router.post("/api/v1/base/inicializar")
async def iniciar_processamento_lote(collection: str):
    """
    Dispara o processamento em lote via bucket de forma assíncrona e retorna um task_id.
    """
    if not collection:
        raise HTTPException(status_code=400, detail="Informe a collection de destino para disparar o lote.")
    
    # Gera um identificador curto para a tarefa (ex: '8f3b2a1c')
    task_id = str(uuid.uuid4())[:8]
    
    TAREFAS_STATUS[task_id] = {
        "status": "pendente",
        "mensagem": "Requisição recebida; a tarefa aguarda o início da execução...",
        "collection": collection
    }
    
    # Inicia a execução em segundo plano usando Threading
    thread = threading.Thread(target=executar_lote_background, args=(task_id, collection))
    thread.start()
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "aceito",
            "mensagem": "Execução em lote assumida por uma thread de fundo.",
            "task_id": task_id
        }
    )

@router.get("/api/v1/tarefas")
async def consultar_status_tarefa(task_id: str):
    """
    Retorna o status atual da tarefa usando query parameter (?task_id=XYZ).
    """
    if not task_id:
        raise HTTPException(status_code=400, detail="Informe o identificador da tarefa (task_id).")
        
    if task_id not in TAREFAS_STATUS:
        raise HTTPException(status_code=404, detail="Nenhuma tarefa corresponde ao ID informado.")
    
    return JSONResponse(
        status_code=200,
        content=TAREFAS_STATUS[task_id]
    )