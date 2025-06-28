# admin.py

import streamlit as st
import fitz
import pandas as pd
import google.generativeai as genai
from faiss_utils import (
    carregar_index,
    adicionar_documento,
    remover_documento_por_nome
)

API_KEY = "SUA_CHAVE_AQUI"
genai.configure(api_key=API_KEY)

INDEX_PATH = "data/indexes/rag.index"
META_PATH = "data/metadados/rag_meta.json"

def extrair_texto_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return [page.get_text() for page in doc]

def extrair_texto_txt(file):
    return [file.read().decode("utf-8", errors="ignore")]

def extrair_texto_excel(file):
    df = pd.read_excel(file)
    return [df.to_string(index=False)]

def chunk_fixo(texto, tamanho, sobreposicao):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunks.append(texto[inicio:fim])
        if fim >= len(texto): break
        inicio += tamanho - sobreposicao
    return chunks

def chunk_por_sinal(texto, delimitador):
    return [p.strip() for p in texto.split(delimitador) if p.strip()]

def chunk_por_pagina(paginas, n_paginas):
    chunks = []
    for i in range(0, len(paginas), n_paginas):
        bloco = "\n".join(paginas[i:i+n_paginas])
        chunks.append(bloco)
    return chunks

def chunk_por_linha_excel(linhas, linhas_por_chunk):
    chunks = []
    for i in range(0, len(linhas), linhas_por_chunk):
        bloco = "\n".join(linhas[i:i+linhas_por_chunk])
        chunks.append(bloco)
    return chunks

st.set_page_config(page_title="Painel Admin RAG", layout="wide")
st.title("🛠️ Painel Administrativo - RAG")

index, metadados = carregar_index(INDEX_PATH, META_PATH)

st.header("📤 Upload e Indexação de Documentos")
arquivos = st.file_uploader("Envie arquivos PDF, TXT ou Excel", type=["pdf", "txt", "xlsx"], accept_multiple_files=True)

metodo = st.selectbox("Método de chunking:", [
    "Chunk por tamanho fixo",
    "Chunk por sinal",
    "Chunk por página",
    "Chunk por documento inteiro",
    "Chunk por linhas (Excel)"
])

chunk_args = {}
if metodo == "Chunk por tamanho fixo":
    chunk_args['tamanho'] = st.number_input("Tamanho do chunk (nº caracteres)", min_value=1, value=1000, step=1)
    chunk_args['sobreposicao'] = st.number_input("Sobreposição (nº caracteres)", min_value=0, value=200, step=1)
elif metodo == "Chunk por sinal":
    chunk_args['sinal'] = st.text_input("Delimitador (ex: '.', '###')", ".")
elif metodo == "Chunk por página":
    chunk_args['n_paginas'] = st.number_input("Nº de páginas por chunk", min_value=1, value=1, step=1)
elif metodo == "Chunk por linhas (Excel)":
    chunk_args['linhas_por_chunk'] = st.number_input("Nº de linhas por chunk", min_value=1, value=20, step=1)

if metodo and arquivos:
    if st.button("📦 Criar chunks e indexar"):
        for arquivo in arquivos:
            nome = arquivo.name
            tipo = nome.split(".")[-1].lower()
            st.write(f"🔄 Processando: {nome}")

            if tipo == "pdf":
                paginas = extrair_texto_pdf(arquivo)
                if metodo == "Chunk por página":
                    chunks = chunk_por_pagina(paginas, chunk_args['n_paginas'])
                else:
                    texto = "\n".join(paginas)
            elif tipo == "txt":
                texto = extrair_texto_txt(arquivo)[0]
            elif tipo == "xlsx":
                texto = extrair_texto_excel(arquivo)[0]

            if metodo == "Chunk por tamanho fixo":
                chunks = chunk_fixo(texto, chunk_args['tamanho'], chunk_args['sobreposicao'])
            elif metodo == "Chunk por sinal":
                chunks = chunk_por_sinal(texto, chunk_args['sinal'])
            elif metodo == "Chunk por documento inteiro":
                chunks = [texto]
            elif metodo == "Chunk por linhas (Excel)":
                linhas = texto.splitlines()
                chunks = chunk_por_linha_excel(linhas, chunk_args['linhas_por_chunk'])

            chunks = [f"[Fonte: {nome}]\n{c}" for c in chunks]
            adicionar_documento(chunks, nome, index, metadados, INDEX_PATH, META_PATH)

        st.success("✅ Todos os arquivos foram indexados com sucesso!")
        st.experimental_rerun()

st.header("📄 Documentos Indexados")
docs = sorted(set(m.get("origem", "desconhecido") for m in metadados))
if docs:
    doc = st.selectbox("Escolha um documento para remover:", docs)
    if st.button("🗑️ Remover documento"):
        metadados = remover_documento_por_nome(index, metadados, doc, INDEX_PATH, META_PATH)
        st.success(f"Documento '{doc}' removido.")
        st.experimental_rerun()
else:
    st.info("Nenhum documento indexado ainda.")
