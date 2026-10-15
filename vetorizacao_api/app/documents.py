import uuid
import base64
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.pipeline import GraphRagPipeline
from app.tasks import TAREFAS_STATUS

router = APIRouter(tags=["Gerenciamento de Documentos e Bucket"])

class SincronizarBucketPayload(BaseModel):
    collection: str

def executar_sincronizacao_bucket_background(task_id: str, collection: str):
    """
    Executa em background:
    1. Varre a pasta 'documentos/' no ObjectStore configurado.
    2. Processa, vetoriza e une os PDFs ao grafo histórico da collection informada.
    3. Atualiza o JSON unificado no ObjectStore e os registros no repositório do grafo.
    """
    try:
        TAREFAS_STATUS[task_id] = {
            "status": "processando",
            "mensagem": f"Varrendo o diretório 'documentos/' em direção à collection '{collection}'...",
            "collection": collection
        }
        
        # Instancia o pipeline passando a collection de destino desejada
        pipeline = GraphRagPipeline(collection_destino=collection)
        
        def status_callback(msg: str):
            TAREFAS_STATUS[task_id]["mensagem"] = msg
            
        # Executa o método nativo que lê a pasta 'documentos/', processa e faz o merge com o grafo existente
        resultado = pipeline.processar_lote_do_bucket(status_callback=status_callback)
        
        TAREFAS_STATUS[task_id] = {
            "status": "concluido",
            "mensagem": f"Integração finalizada: {resultado['total_docs_processados']} arquivos PDF incorporados ao grafo.",
            "total_arquivos_processados": resultado['total_docs_processados'],
            "total_novos_chunks_gerados": resultado['total_nos'],
            "collection_afetada": collection
        }
        
    except Exception as e:
        TAREFAS_STATUS[task_id] = {
            "status": "erro",
            "mensagem": f"Falha ao sincronizar o bucket: {str(e)}",
            "collection": collection
        }

@router.post("/api/v1/documentos/sincronizar-bucket")
async def endpoint_sincronizar_bucket(payload: SincronizarBucketPayload):
    """
    Endpoint POST que dispara a varredura automática da pasta 'documentos/' do bucket 
    integrando os arquivos ao grafo da collection enviada no JSON.
    """
    if not payload.collection:
        raise HTTPException(status_code=400, detail="O campo 'collection' precisa ser informado.")
    
    task_id = str(uuid.uuid4())[:8]
    
    TAREFAS_STATUS[task_id] = {
        "status": "pendente",
        "mensagem": "Varredura do bucket registrada e aguardando execução...",
        "collection": payload.collection
    }
    
    # Inicia o processo em segundo plano para proteger contra timeout no Gateway
    thread = threading.Thread(target=executar_sincronizacao_bucket_background, args=(task_id, payload.collection))
    thread.start()
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "aceito",
            "mensagem": f"Coleta dos PDFs de 'documentos/' disparada para a collection '{payload.collection}'.",
            "task_id": task_id
        }
    )

class UploadDocumentoPayload(BaseModel):
    base64_file: str
    nome_arquivo: str

@router.post("/api/v1/documentos/upload")
async def endpoint_upload_documento_bucket(payload: UploadDocumentoPayload):
    """
    Recebe um documento em base64 e salva fisicamente no bucket na pasta 'documentos/'.
    Não altera o Banco de Dados e não requer especificar 'collection'.
    """
    if not payload.base64_file or not payload.nome_arquivo:
        raise HTTPException(status_code=400, detail="Envie 'base64_file' e 'nome_arquivo' para concluir o upload.")
    
    try:
        # Decodifica base64
        base64_data = payload.base64_file.strip()
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1].strip()
        base64_data = base64_data.replace(" ", "+")
        pdf_bytes = base64.b64decode(base64_data)
        
        # Validação do cabeçalho do PDF
        if b"%PDF" not in pdf_bytes[:1024]:
            raise HTTPException(status_code=400, detail="O conteúdo decodificado não corresponde a um PDF válido.")

        # Usamos uma collection genérica/padrão apenas para inicializar o pipeline
        pipeline = GraphRagPipeline(collection_destino="default")
        object_name = pipeline.fazer_upload_pdf_bucket(pdf_bytes, payload.nome_arquivo)

        return JSONResponse(
            status_code=201,
            content={
                "status": "sucesso",
                "mensagem": f"'{payload.nome_arquivo}' gravado em 'documentos/' dentro do bucket.",
                "caminho_bucket": object_name
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao gravar o arquivo no bucket: {str(e)}")

@router.get("/api/v1/documentos")
async def endpoint_listar_documentos_collection(collection: str):
    """
    Retorna a lista de documentos únicos integrados ao grafo da collection especificada.
    """
    if not collection:
        raise HTTPException(status_code=400, detail="Informe o parâmetro 'collection' para a consulta.")
    
    try:
        pipeline = GraphRagPipeline(collection_destino=collection)
        nodes = pipeline.recuperar_historico_grafo()
        
        # Coleta doc_id únicos dos nós do grafo
        documentos = sorted(list({n.get("doc_id") for n in nodes if n.get("doc_id")}))
        
        return JSONResponse(
            status_code=200,
            content={
                "collection": collection,
                "total_documentos": len(documentos),
                "documentos": documentos
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Não foi possível recuperar os documentos da collection '{collection}': {str(e)}")

@router.get("/api/v1/bucket/documentos")
async def endpoint_listar_documentos_bucket():
    """
    Lista todos os PDFs fisicamente presentes na pasta 'documentos/' do bucket.
    Não requer especificar 'collection'.
    """
    try:
        # Usamos uma collection genérica/padrão apenas para inicializar o pipeline
        pipeline = GraphRagPipeline(collection_destino="default")
        arquivos = pipeline.listar_arquivos_pdf_bucket()
        return JSONResponse(
            status_code=200,
            content={
                "status": "sucesso",
                "total_arquivos": len(arquivos),
                "arquivos": arquivos
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao consultar os arquivos do bucket: {str(e)}")

@router.delete("/api/v1/bucket/documentos/limpar")
async def endpoint_limpar_bucket(file_name: str = None):
    """
    Limpa a pasta 'documentos/' do bucket.
    Se 'file_name' for fornecido, remove apenas aquele arquivo do bucket.
    Se 'file_name' não for fornecido, limpa todos os PDFs da pasta.
    Não altera o Banco de Dados e não requer especificar 'collection'.
    """
    try:
        # Usamos uma collection genérica/padrão apenas para inicializar o pipeline
        pipeline = GraphRagPipeline(collection_destino="default")
        total_removidos = pipeline.limpar_pasta_pdf_bucket(file_name=file_name)
        
        msg = f"'{file_name}' excluído do bucket." if file_name else f"Diretório 'documentos/' esvaziado; {total_removidos} arquivo(s) retirado(s) do bucket."
        return JSONResponse(
            status_code=200,
            content={
                "status": "sucesso",
                "mensagem": msg,
                "total_arquivos_removidos": total_removidos
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao limpar o bucket: {str(e)}")
