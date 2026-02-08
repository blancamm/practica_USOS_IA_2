# Como visto en clase instalamos todas las dependecias necesarias de propias de texto, directorio y framework como langchain y transofrmer:
import os
import re
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans # se leige modelo de ML para hacer la agrupación por clusters

# Todos estos es lo que se ha indicado en el documento principal, que sirven como framework para hacer el proceso RGA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import nltk
nltk.download("stopwords") # se usa luego para limpiar el texto de los PDFs
from nltk.corpus import stopwords
from collections import Counter

# ===============================CONFIGURACION=====================
#Configuramos todo el directorio de entrada y salida, y decidimos los modelos y parametros del pipline como el numero de chucks a tener en cuenta

DATA_DIR = "data_entrada"
OUT_DIR = "salidas"

# Los modelos elegidos para convertir texto a vectores (embeddings) y para hacer las preguntas, respectivamente:
EMBEDDINGS_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "google/flan-t5-base"

# se decide e cuanto trocear, y en cuantos cluster min y max se dividira cada PDF
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

KMIN = 3
KMAX = 9
# se decide el numero maximo de keywords por cada cluster
TOP_KEYWORDS = 6
EXAMPLES_PER_TOPIC = 6


# ===================== ERRORES ================================0
#Segun se ha ido probrando el modelo, se ha planteado la posibilidad de que ya hubiese estando "transformados"los PDFs en mapa mental
#por lo que si es así, no peta el programa y no se fuerza a volver a hacerse
FORCE_REBUILD = False

#Se ha intentado exportar los mapas metales a PDF. Esto no lo he conseguido por mi version de NOde, puede que otro usuario le valga
EXPORT_PDF = True


# ====================CREACION DE FUNCIONES PARA MANEJAR EL INICIO Y EL FINAL  ===========
@dataclass
class ChunkItem:
    """Trozo de texto con metadatos mínimos."""
    text: str
    source: str
    page: int

# Paso 1: se chequea que la carpeta de salida existe para poner ahi los mapas mentales
def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

# Paso 2: se estudia la carpeta de entrada, viendo la lista de pdfs
def list_pdf_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise RuntimeError(f"No existe la carpeta de entrada: {folder}")

    pdfs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]
    return sorted(pdfs)

# Paso 3: Para coger de cada PDF su nombre sin la extension para ponerselo a su mapa mental correspondiente
def safe_stem_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", base).strip()
    base = re.sub(r"\s+", "_", base)
    return base or "documento"

# =================================================================================

# ==================== FUNCIONES PARA TRABAJAR CON EL TEXTO Y CREAR DEL MAPA MENTAL ============

# Normaliza el texto de entrada: quita epsacios de la entrada, medio y del final (hecho en el modelo NPL)
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

#Crear las etiquetas del mapa mental en función de las keywords. Se usa más tarde
def sanitize_mermaid_label(label: str) -> str:
    label= label.replace('"', "")
    label = label.replace("[", "").replace("]", "")
    label= label.replace(":", "")
    label = label.replace("(", "").replace(")", "")
    label = re.sub(r"\s+", " ", label).strip()
    return label

# 1º paso importante: trocear el pdf en chunks (se hace segun framework langchain):
def load_pdf_as_chunks(pdf_path: str) -> List[ChunkItem]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len)

    # se carga el pdf, se convierte a doc y se divide
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    items: List[ChunkItem] = []
    for c in chunks:
        page = c.metadata.get("page", -1)
        items.append(
            ChunkItem(
                text=clean_text(c.page_content),  #<- se crea una objeto chunkitem por cada pdf
                source=os.path.basename(pdf_path),
                page=int(page) if page is not None else -1)
        )
    return items


#2ª Una vez creado los chuncks se pasan a vector numerico (embedding) con el modelo elegido al principio
def embed_chunks(chunks: List[ChunkItem]) -> np.ndarray:
    emb = HuggingFaceEmbeddings(model_name= EMBEDDINGS_MODEL)
    vectors = emb.embed_documents([c.text for c in chunks])
    return np.array(vectors, dtype=np.float32)

# 3º En funcion de los parametros definif¡dos al principio se eleigen entre los chunks creados.
#Es cierto que estos parametros habría que tal vez ajustarlo dependiendo del tamaño del pdf, usando otras metricas
# De momento se hace el siguiente razonamiento: si hay muchos chunks en genera, hay mas temas, y si en total, son menos, pues se utilizan el minimo numero establecido al principio
def choose_k(n_chunks: int) -> int:
    if n_chunks < 20:
        return KMIN
    if n_chunks< 60:
        return min(5, KMAX)
    if n_chunks < 120:
        return min(7,KMAX)
    return KMAX

# 4º Paso: se busca entre los chunks, los que deben ser los elegidos para el mapa mental
# para ellos se hace primero una limpieza (se va comentando en las lineas imporantes):
def extract_keywords_simple(texts: List[str], top_n: int = TOP_KEYWORDS) -> List[str]:
    # se define una lista de stopwords como veiamos en el modulo NPL anterior (es la descragada)
    stop_words = set(stopwords.words("spanish"))

    words: List[str] = []
    for t in texts:
        #Con lo siguiente normalizas el texto, quitando acentos que no froman parte del castellano
        t = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9 ]", " ", t)
        for w in t.lower().split():
            if len(w) < 4:
                continue
            if w in stop_words:
                continue
            if w.isdigit(): #eliminas los numeros
                continue
            words.append(w)

    if not words:
        return ["tema"]

    from collections import Counter
    c = Counter(words)
    return [w for w, _ in c.most_common(top_n)] # de esta manera, cogiendo las mas similares al tema, eliges las palabras del mapa mental


# 5º Se contruye  el mapa mental:
def build_mindmap(chunks: List[ChunkItem], vectors: np.ndarray, root_title: str) -> Dict:
    n= len(chunks)
    k= choose_k(n)
    
    #se agrupan por temaS en funcion de minima distancia con el modelo del modulo de ML, CREANDO CLUSTERS
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(vectors) # Se predice a que tema corresponde cada k previamente elegido

    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    mind = {"root": root_title, "topics": []}

    for _, idxs in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        cluster_texts = [chunks[i].text for i in idxs]
        keywords = extract_keywords_simple(cluster_texts, top_n=TOP_KEYWORDS) #se sacan las keywords para el mapa segun los clusters hechos

        title = " / ".join(keywords[:3]) if keywords else "Tema" #se saca la principal de cada cluster 

        examples = [] # se guardan el resto para crear las ramas del mapa
        for i in idxs[: min(EXAMPLES_PER_TOPIC, len(idxs))]:
            snippet = chunks[i].text[:200]
            if len(chunks[i].text) > 200:
                snippet += "..."
            examples.append({
                "source": chunks[i].source,
                "page": chunks[i].page,
                "snippet": snippet})

        mind["topics"].append({
            "title": title,
            "keywords": keywords,
            "size": len(idxs),
            "examples": examples})

    return mind

# =======================================================================================

# ============================== CREACION DE LOS OUTPUTS ===============================

#Se guarda en Json, y se sobreescribe si está
def export_json(mind: Dict, out_json_path: str) -> None:
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(mind, f, ensure_ascii=False, indent=2)

#Se exporta a este otro tipo de archivo que es 'mermaid', el cual antes no he conocía y he investigado, pero se necesita pagina web para verlo
def export_mermaid_mindmap(mind: Dict, out_mmd_path: str) -> None:
    root = sanitize_mermaid_label(mind["root"]) # se utiliza la función previamente creada apra las labels
    lines = ["mindmap", f"  root(({root}))"]

    for t in mind["topics"]:
        topic_label = f"{t['title']} ({t['size']})"
        topic_label = sanitize_mermaid_label(topic_label)
        lines.append(f"    {topic_label}") # <- se deja los espacios para indicar la jerarquiar segun como lo hace MERmaid

        for kw in t["keywords"][:TOP_KEYWORDS]:
            kw =sanitize_mermaid_label(kw)
            lines.append(f"      {kw}")

    with open(out_mmd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# se chequea si existe el archivo
def mermaid_cli_available() -> bool:
    return shutil.which("mmdc") is not None

#Convierte .mmd -> .pdf usando mermaid-cli (mmdc). A mi no me funciona, pero puede que al profesor con otros nodes, sí
# se ha corregido su error, para que si no funciona, el codigo no colapse, que me pasó
def export_pdf_with_mmdc(mmd_path: str, pdf_path: str) -> bool:
    if not mermaid_cli_available():
        return False

    cmd =["mmdc", "-i", mmd_path, "-o", pdf_path]
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        # mmdc no está disponible en PATH o está mal instalado
        return False
    except subprocess.CalledProcessError:
        # mmdc existe pero falló (node/puppeteer, etc.)
        return False

# En un control de errores, se tuvo que crear esta funcion, para que si ya esta hechos los archivos, y se vuelve a ejecutar el scrip no colapse.
def outputs_exist(out_json: str, out_mmd: str, out_pdf: str, want_pdf: bool) -> bool:
    if not (os.path.isfile(out_json) and os.path.isfile(out_mmd)):
        return False
    if want_pdf and not os.path.isfile(out_pdf):
        return False
    return True

# =======================================================================================================0
# =======================================================================================================0

# ====================================== CAPA QA =========================================================0
# Para hacer algo más parecido a lo visto en clase, se añadió esta capa simple de preguntas sobre los PDFs procesado.
# Como se comenta en el docuemnto, al no tener api key, no se utilizan métodos tan complejos como los vistos en clase (como chatgpt)
def build_local_rag(chunks: List[ChunkItem]):
    from langchain_core.documents import Document # se pone aqui por si no pregunta, para que no se importe innecesariamenre

    docs = []
    for c in chunks:
        docs.append(Document(page_content=c.text,metadata={"source": c.source, "page": c.page}))

    #como visto en clase
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vs = FAISS.from_documents(docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    hf_pipe = pipeline(
        task="text2text-generation",
        model=LOCAL_LLM_MODEL,
        max_new_tokens=220,
        device=-1
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    prompt = ChatPromptTemplate.from_template(
        """Responde usando SOLO el contexto. Si no está, di: "No aparece en los documentos".

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""
    )

    def format_docs(ds):
        return "\n\n".join([
            f"[{d.metadata.get('source')} p.{d.metadata.get('page')}] {d.page_content}"
            for d in ds
        ])

    chain = ({"context": retriever | format_docs, "question": lambda x: x} | prompt | llm | StrOutputParser())
    return chain


# ============================================================
# ============================================================

# ============================ FUNCIONES PRINCIPALES PARA EJECUTAR =========================

# Llamamos a todas las funciones que hemos ido creando:

def process_one_pdf(pdf_path: str, export_pdf: bool = True, force: bool = False) -> Dict[str, str]:
    base = safe_stem_filename(pdf_path)

    out_json= os.path.join(OUT_DIR, f"{base}_mindmap.json")
    out_mmd= os.path.join(OUT_DIR, f"{base}_mindmap.mmd")
    out_pdf= os.path.join(OUT_DIR, f"{base}_mindmap.pdf")

    # Si ya existe salida y no forzamos, saltamos para evitar reprocesar
    if (not force) and outputs_exist(out_json, out_mmd, out_pdf, want_pdf=export_pdf):
        return {
            "pdf": pdf_path,
            "json": out_json,
            "mmd": out_mmd,
            "pdf_out": out_pdf if (export_pdf and os.path.isfile(out_pdf)) else "",
            "status": "skipped"
        }

    #Siguiendo los pasos mostrados y comentados en el documento
    #1º Extraer chunks
    chunks= load_pdf_as_chunks(pdf_path)

    #2ª Crear los embeddings
    vectors = embed_chunks(chunks)

    # 3º Crear mapa mental
    root_title = f"Mapa mental - {base}"
    mind = build_mindmap(chunks, vectors, root_title=root_title)

    # 4º Exportar el mapa mental en formato json y mermaid 
    export_json(mind, out_json)
    export_mermaid_mindmap(mind, out_mmd)

    #5º Crear eñ PDF (si se pidió y si se puede)
    pdf_done = False
    if export_pdf:
        pdf_done = export_pdf_with_mmdc(out_mmd, out_pdf)

    return {
        "pdf": pdf_path,
        "json": out_json,
        "mmd": out_mmd,
        "pdf_out": out_pdf if pdf_done else "",
        "status": "built"}


def main():
    ensure_dirs()

    pdf_paths = list_pdf_files(DATA_DIR)
    if not pdf_paths:
        raise RuntimeError(f"No hay PDFs en {DATA_DIR}. Añade PDFs y vuelve a ejecutar.")

    print("Los PDFs que se van a procesar son:")
    for p in pdf_paths:
        print(" -",  p)

    results = []
    for pdf in pdf_paths:
        res =process_one_pdf(pdf, export_pdf= EXPORT_PDF, force= FORCE_REBUILD)
        results.append(res)

        print(f"Procesando: {os.path.basename(pdf)}")
        print("  - JSON:", res["json"])
        print("  - MMD :", res["mmd"])
        if EXPORT_PDF:
            if res["pdf_out"]:
                print("  - PDF :", res["pdf_out"])
            else:
                print("  - PDF : (no generado; instala mermaid-cli si lo necesitas)")
        print("  - Estado:", res["status"])
        print()

    # Activar según respuesta el modo QA opcional -> se pregunta si quiere entrar. Si dice s o si, se entrar, todo lo contrario se sale
    print("¿Quieres activar modo preguntas (RAG local) sobre un PDF? (s/n)")
    ans = input("> ").strip().lower()
    if (ans != "s") and (ans != "si"):
        return

    print("\nSelecciona un PDF por número:")
    for i, p in enumerate(pdf_paths, 1):
        print(f"  {i}. {os.path.basename(p)}")

    sel = input("> ").strip()
    if not sel.isdigit():
        return

    idx = int(sel) - 1
    if idx < 0 or idx >= len(pdf_paths):
        return

    selected_pdf = pdf_paths[idx]
    print(f"\nCargando para QA: {os.path.basename(selected_pdf)}")

    chunks = load_pdf_as_chunks(selected_pdf)
    rag = build_local_rag(chunks)

    print("\nModo QA. Escribe 'salir' para terminar.\n")
    while True:
        q = input("Tú: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        if not q:
            continue
        print("IA:", rag.invoke(q))
        print()


if __name__ == "__main__":
    main()