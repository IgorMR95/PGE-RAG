import os
import json
import faiss
import numpy as np

DIM = 384  # deve ser igual ao dim retornado pelo modelo de embedding

def criar_index():
    """
    Cria um índice FAISS vazio com suporte a IDs.
    """
    index = faiss.IndexFlatIP(DIM)
    return faiss.IndexIDMap(index)

def salvar_index(index, path_index):
    """
    Salva o índice FAISS em disco.
    """
    os.makedirs(os.path.dirname(path_index), exist_ok=True)
    faiss.write_index(index, path_index)

def salvar_metadados(metadados, path_meta):
    """
    Salva os metadados em JSON no disco.
    """
    os.makedirs(os.path.dirname(path_meta), exist_ok=True)
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

def carregar_index(path_index, path_meta):
    """
    Carrega índice FAISS e metadados do disco. Se não existirem, cria novos.
    """
    # Carrega ou cria índice
    if os.path.exists(path_index):
        index = faiss.read_index(path_index)
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
    else:
        index = criar_index()

    # Carrega ou cria metadados
    if os.path.exists(path_meta):
        try:
            with open(path_meta, "r", encoding="utf-8") as f:
                metadados = json.load(f)
        except json.JSONDecodeError:
            metadados = []
    else:
        metadados = []

    return index, metadados

def adicionar_documento(chunks, origem, index, metadados, path_index, path_meta):
    """
    Adiciona chunks ao índice e atualiza os metadados.

    chunks: lista de strings, cada uma representando um chunk.
    origem: nome do arquivo ou identificador do documento.
    """
    if not chunks:
        print("⚠️ Nenhum chunk fornecido.")
        return

    from embedder import gerar_embeddings_para_chunks
    embeddings = gerar_embeddings_para_chunks(chunks)

    if len(embeddings.shape) != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("❌ Número de embeddings não corresponde ao número de chunks.")

    # Gerar novos IDs
    ids_existentes = {m["id"] for m in metadados if "id" in m}
    proximo_id = max(ids_existentes) + 1 if ids_existentes else 0
    ids_novos = np.arange(proximo_id, proximo_id + len(chunks)).astype(np.int64)

    # Adiciona ao índice
    index.add_with_ids(embeddings, ids_novos)

    # Atualiza metadados
    for i, chunk in enumerate(chunks):
        metadados.append({
            "id": int(ids_novos[i]),
            "origem": origem,
            "texto": chunk
        })

    salvar_index(index, path_index)
    salvar_metadados(metadados, path_meta)

def buscar_similares(embedding_query, index, metadados, k=5):
    """
    Busca os k chunks mais similares a uma query.
    Retorna metadados correspondentes.
    """
    if index.ntotal == 0:
        return []

    embedding_query = np.array([embedding_query]).astype("float32")
    _, ids = index.search(embedding_query, k)

    resultados = []
    for idx in ids[0]:
        if idx == -1:
            continue
        md = next((m for m in metadados if m["id"] == int(idx)), None)
        if md:
            resultados.append(md)
    return resultados

def remover_documento_por_nome(index, metadados, nome_arquivo, path_index, path_meta):
    """
    Remove todos os chunks de um documento do índice e dos metadados.
    """
    ids_remover = [m["id"] for m in metadados if m.get("origem") == nome_arquivo]
    if not ids_remover:
        print(f"⚠️ Nenhum chunk encontrado para '{nome_arquivo}'")
        return metadados

    index.remove_ids(np.array(ids_remover).astype(np.int64))
    metadados = [m for m in metadados if m.get("origem") != nome_arquivo]

    salvar_index(index, path_index)
    salvar_metadados(metadados, path_meta)
    return metadados
