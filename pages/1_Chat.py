# chat.py

import streamlit as st
import google.generativeai as genai
from faiss_utils import carregar_index, buscar_similares
from embedder import gerar_embedding

# Configuração do modelo
API_KEY = "AIzaSyBrRaTvJ6wo2-PQUxOwRMQzr1S8KucT79A"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

INDEX_PATH = "data/indexes/rag.index"
META_PATH = "data/metadados/rag_meta.json"

# Carrega o índice e os metadados
index, metadados = carregar_index(INDEX_PATH, META_PATH)

st.set_page_config(page_title="Consultor Normativo do Contencioso Geral da PGE-SP", layout="wide")
st.title("📚 Consultor Normativo do Contencioso Geral da PGE-SP")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Chat
query = st.chat_input("Digite sua pergunta:")
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query:
    st.session_state.mensagens.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    query_vector = gerar_embedding(query)
    context_chunks = buscar_similares(query_vector, index, metadados, k=5)
    contexto = "\n\n".join([c["texto"] for c in context_chunks if c])

    prompt = f"""Você é um advogado especialista. Use as fontes abaixo para responder as dúvidas e sempre que responder aponte qual foi a fonte de onde voce tirou a informação. Ao citar a fonte cite o nome do arquivo mas não precisa citar a extensão (Exemplo, se a fonte foi o arquivo "teste.pdf" diga apenas que a fonte da citação foi "teste". Seja literal na interpretação dos materiais, ou seja, não faça interpretações implícitas ou expansivas sobre os conteúdos.

Fontes:
{contexto}

Pergunta:
{query}
"""
    resposta = model.generate_content(prompt).text.strip()
    st.session_state.mensagens.append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)
