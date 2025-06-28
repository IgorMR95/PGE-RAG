import os
import fitz  # PyMuPDF
from faiss_utils import carregar_index, adicionar_documento

# Caminhos do índice e dos metadados
INDEX_PATH = "data/indexes/rag.index"
META_PATH = "data/metadados/rag_meta.json"

# Configurações de chunking
TAMANHO_CHUNK = 1000       # número de caracteres por chunk
SOBREPOSICAO = 200         # número de caracteres que se sobrepõem entre chunks

# Pasta onde estão os arquivos a indexar
PASTA_DOCUMENTOS = "data/documentos"

# Função para extrair texto de PDF
def extrair_texto_pdf(caminho_arquivo):
    with fitz.open(caminho_arquivo) as doc:
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
    return texto

# Função para extrair texto de TXT
def extrair_texto_txt(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# Carrega (ou cria) o índice e os metadados
index, metadados = carregar_index(INDEX_PATH, META_PATH)

# Processa todos os arquivos da pasta
for nome_arquivo in os.listdir(PASTA_DOCUMENTOS):
    caminho_arquivo = os.path.join(PASTA_DOCUMENTOS, nome_arquivo)

    if not os.path.isfile(caminho_arquivo):
        continue

    ext = nome_arquivo.split(".")[-1].lower()
    print(f"🔄 Indexando: {nome_arquivo}")

    if ext == "pdf":
        texto = extrair_texto_pdf(caminho_arquivo)
    elif ext == "txt":
        texto = extrair_texto_txt(caminho_arquivo)
    else:
        print(f"❌ Formato não suportado: {nome_arquivo}")
        continue

    # Adiciona ao índice com chunking configurável
    adicionar_documento(
        texto,
        nome_arquivo,
        index,
        metadados,
        INDEX_PATH,
        META_PATH,
        tamanho_chunk=TAMANHO_CHUNK,
        sobreposicao=SOBREPOSICAO
    )

print("✅ Indexação finalizada com sucesso.")
