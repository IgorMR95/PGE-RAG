from sentence_transformers import SentenceTransformer
import numpy as np

# Modelo carregado apenas uma vez por instância
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo de embeddings: {e}")

def gerar_embedding(texto):
    """
    Gera embedding para um único texto.
    
    Parâmetros:
        texto (str): Texto para gerar o embedding.
    
    Retorno:
        np.ndarray: vetor (dim,) do embedding em float32.
    """
    if not isinstance(texto, str):
        raise ValueError("O texto deve ser uma string.")
    
    embedding = model.encode(texto)
    return embedding.astype('float32')

def gerar_embeddings_para_chunks(chunks):
    """
    Gera embeddings para uma lista de strings (chunks).
    
    Parâmetros:
        chunks (list of str): Lista de textos.
    
    Retorno:
        np.ndarray: matriz (N, D) de embeddings em float32.
    """
    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        raise ValueError("A entrada deve ser uma lista de strings.")

    embeddings = model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings).astype('float32')
