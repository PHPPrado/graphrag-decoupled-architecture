import base64
import uuid
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.pipeline import GraphRagPipeline
from app.tasks import TAREFAS_STATUS

router = APIRouter(tags=["Gerenciamento de Documentos"])

class SobrescreverDocumentoPayload(BaseModel):
    doc_id: str
    collection: str
    base64_file: str
    nome_arquivo: str = "documento.pdf"

def executar_sobrescrita_background(task_id: str, payload: SobrescreverDocumentoPayload):
    """
    Executa em background:
    1. Remove o documento antigo do banco e do grafo (cobrindo variações de prefixo).
    2. Decodifica o base64 e processa em memória (sem gravar em disco).
    3. Une os novos nós com o restante do grafo e salva no Object Storage.
    """
    try:
        TAREFAS_STATUS[task_id] = {
            "status": "processando",
            "mensagem": f"Substituição do documento '{payload.doc_id}' em andamento...",
            "collection": payload.collection
        }
        
        pipeline = GraphRagPipeline(collection_destino=payload.collection)
        
        # --- BLINDAGEM DO DOC_ID (Com e sem prefixo 'documentos/') ---
        doc_alvo = payload.doc_id.strip()
        doc_alvo_com_prefixo = doc_alvo if doc_alvo.startswith("documentos/") else f"documentos/{doc_alvo}"
        doc_alvo_sem_prefixo = doc_alvo.replace("documentos/", "")
        
        # --- PASSO 1: IDENTIFICAÇÃO E REMOÇÃO DO ANTIGO ---
        TAREFAS_STATUS[task_id]["mensagem"] = "Lendo o histórico do grafo para mapear os nós do documento alvo..."
        
        # 1. Recupera o histórico do grafo PRIMEIRO
        historico_atual = pipeline.recuperar_historico_grafo()

        # 2. Identifica os nós do documento alvo e pesca o NOME EXATO gravado
        nodes_alvo = [
            n for n in historico_atual 
            if n.get("doc_id") in [doc_alvo, doc_alvo_com_prefixo, doc_alvo_sem_prefixo]
        ]
        
        # 3. Pega a identidade real do arquivo
        doc_id_alvo_final = nodes_alvo[0].get("doc_id") if nodes_alvo else doc_alvo

        TAREFAS_STATUS[task_id]["mensagem"] = f"Descartando a versão anterior de '{doc_id_alvo_final}' no repositório..."

        # 4. AGORA SIM! Manda o repositório apagar usando o nome real
        pipeline.remover_documento_do_banco(doc_id=doc_id_alvo_final)

        # 5. Limpa do JSON em memória
        nodes_restantes = [
            n for n in historico_atual 
            if n.get("doc_id") != doc_id_alvo_final
        ]
        
        # Reconstrói as arestas internas, removendo as que apontavam para o documento apagado
        ids_restantes = {n["id"] for n in nodes_restantes}
        for node in nodes_restantes:
            node["edges"] = [
                e for e in node.get("edges", []) 
                if e.get("target") in ids_restantes and e.get("source") in ids_restantes
            ]
            
        # --- PASSO 2: PROCESSAMENTO DO NOVO ARQUIVO (BASE64) ---
        TAREFAS_STATUS[task_id]["mensagem"] = f"Convertendo e vetorizando o novo arquivo no lugar dos nós de '{doc_id_alvo_final}'..."
        
        # Limpa e valida o conteúdo base64, removendo possíveis prefixos de Data URL (Ex: data:application/pdf;base64,)
        base64_data = payload.base64_file.strip()
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1].strip()
            
        # Corrige espaços que deveriam ser '+' (comum em transmissões HTTP/JSON quando não codificados na URL)
        base64_data = base64_data.replace(" ", "+")
            
        pdf_bytes = base64.b64decode(base64_data)
        
        # Validação de integridade de cabeçalho PDF para diagnóstico preciso
        if b"%PDF" not in pdf_bytes[:1024]:
            raise Exception(
                f"Assinatura '%PDF' não encontrada nos primeiros bytes decodificados. "
                f"Isso indica que os dados base64 fornecidos não representam um arquivo PDF válido. "
                f"Primeiros 50 bytes decodificados: {pdf_bytes[:50]!r}"
            )

        # Define o NOVO nome! Se veio no payload usa ele, senão usa o doc_id puro.
        nome_final = payload.nome_arquivo.strip() if payload.nome_arquivo else doc_alvo_sem_prefixo
        
        def status_callback(msg: str):
            TAREFAS_STATUS[task_id]["mensagem"] = msg
            
        # IMPORTANTE: Passando nome_final para que o arquivo receba o nome novo desejado
        novos_nodes = pipeline.processar_novo_documento(
            pdf_bytes=pdf_bytes, 
            file_name=nome_final, # <--- AGORA SIM USA O NOME NOVO
            historico_grafo=nodes_restantes,
            status_callback=status_callback
        )
        
        if not novos_nodes:
            raise Exception("O processamento não gerou nenhum chunk. O PDF pode estar vazio, corrompido, ou ser apenas uma imagem sem texto (OCR não aplicado nesta rota).")
        
        # --- PASSO 3: UNIÃO E RECONSTRUÇÃO DO GRAFO ---
        TAREFAS_STATUS[task_id]["mensagem"] = "Mesclando o conteúdo atualizado ao grafo e persistindo no ObjectStore..."
        
        # Junta o que sobrou do grafo antigo com os novos nós gerados
        nodes_totais = nodes_restantes + novos_nodes
        
        # Reconstrói e salva o grafo unificado no Object Storage
        grafo_url = pipeline.reconstruir_e_salvar_grafo(nodes_totais)
        
        TAREFAS_STATUS[task_id] = {
            "status": "concluido",
            "mensagem": f"Substituição concluída: '{payload.doc_id}' agora aponta para '{payload.nome_arquivo}'.",
            "total_nos_grafo": len(nodes_totais),
            "grafo_atualizado": grafo_url,
            "collection_afetada": payload.collection
        }
        
    except Exception as e:
        TAREFAS_STATUS[task_id] = {
            "status": "erro",
            "mensagem": f"Sobrescrita cancelada por erro: {str(e)}",
            "collection": payload.collection
        }

@router.put("/api/v1/documentos/sobrescrever")
async def endpoint_sobrescrever_documento(payload: SobrescreverDocumentoPayload):
    """
    Endpoint PUT para sobrescrever um documento existente via JSON (Base64 + collection + Doc_ID).
    Retorna um task_id para acompanhamento assíncrono.
    """
    if not payload.doc_id or not payload.collection or not payload.base64_file:
        raise HTTPException(status_code=400, detail="Informe 'doc_id', 'collection' e 'base64_file' para continuar.")
    
    task_id = str(uuid.uuid4())[:8]
    
    TAREFAS_STATUS[task_id] = {
        "status": "pendente",
        "mensagem": "Sobrescrita enfileirada, aguardando liberação da thread...",
        "collection": payload.collection
    }
    
    # Dispara a thread em segundo plano
    thread = threading.Thread(target=executar_sobrescrita_background, args=(task_id, payload))
    thread.start()
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "aceito",
            "mensagem": "Substituição assumida por processamento em background.",
            "task_id": task_id
        }
    )