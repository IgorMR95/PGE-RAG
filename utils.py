def chunkear_texto(texto, tamanho_max=500):
    paragrafos = texto.split("\n")
    chunks = []
    buffer = ""

    for p in paragrafos:
        if len(buffer) + len(p) < tamanho_max:
            buffer += p.strip() + " "
        else:
            chunks.append(buffer.strip())
            buffer = p.strip() + " "
    if buffer:
        chunks.append(buffer.strip())

    return [c for c in chunks if len(c) > 50]
