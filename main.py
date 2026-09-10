import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

app = FastAPI(title="PDF Chat API")
app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = os.getenv("API_KEY")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

existing = [c.name for c in qdrant_client.get_collections().collections]
if os.getenv("COLLECTION_NAME") not in existing:
    qdrant_client.create_collection(
        collection_name=os.getenv("COLLECTION_NAME"),
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding=embeddings
)

prompt = ChatPromptTemplate.from_template("""
Answer based only on the context below.
Context: {context}
Question: {question}
""")

chain = (
    {"context": vector_store.as_retriever(search_kwargs={"k": 4}), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), _: None = Depends(verify_api_key)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    text = "".join([page.extract_text() for page in reader.pages])
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    chunks = [Document(page_content=t, metadata={"source": file.filename}) for t in texts]

    vector_store.add_documents(chunks)

    return {"message": f"Uploaded {len(chunks)} chunks", "filename": file.filename}

@app.post("/chat")
async def chat(body: dict, _: None = Depends(verify_api_key)):
    answer = chain.invoke(body["question"])
    return {"answer": answer}

@app.get("/documents")
async def list_documents(_: None = Depends(verify_api_key)):
    results = qdrant_client.scroll(
        collection_name=os.getenv("COLLECTION_NAME"),
        with_payload=True,
        limit=100
    )
    sources = list(set(
        point.payload.get("metadata", {}).get("source", "Unknown")
        for point in results[0]
    ))
    return {"documents": sources}

@app.get("/")
def root():
    return {"status": "PDF Chat API running"}