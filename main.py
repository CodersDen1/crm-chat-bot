import os
import uuid
import shutil
import requests
import asyncio
import re
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from typing_extensions import TypedDict
from contextlib import asynccontextmanager

# Import necessary LangChain components
from langchain import hub
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser

# Import OpenAI LLM and embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Import PDF processing library
import PyPDF2

# -------------------------------
# Environment Setup
# -------------------------------
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# -------------------------------
# Pydantic Models for structured output and API payloads
# -------------------------------
class Search(BaseModel):
    query: str
    section: Literal["beginning", "middle", "end"]


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


class FileInfo(BaseModel):
    file_id: str
    original_filename: str


class GoogleDocRequest(BaseModel):
    url: str


# -------------------------------
# Global Variables and Initialization
# -------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define the static files directory
STATIC_DIR = "static"

# Initialize LLM, Embeddings, and Vector Store
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Using OpenAI's ChatGPT
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Use a text splitter to break file content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# We pull a prompt for question-answering.
prompt = hub.pull("rlm/rag-prompt")

# Create a structured output parser for Search extraction.
structured_parser = PydanticOutputParser(pydantic_object=Search)


# -------------------------------
# Utility Functions
# -------------------------------
def load_all_uploaded_files() -> List[Document]:
    """
    Load and split the text content from all files in the uploads directory.
    Supports text files and PDFs.
    """
    all_docs = []
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.txt', '.md', '.csv', '.json', '.pdf')):
            text = ""
            try:
                if fname.lower().endswith(".pdf"):
                    # Extract text from PDF
                    with open(fpath, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                else:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue
            if text.strip():
                doc = Document(page_content=text, metadata={"file_id": fname})
                splits = text_splitter.split_documents([doc])
                all_docs.extend(splits)
    return all_docs


def reindex_vector_store():
    """
    Rebuild the vector store from the content of all uploaded files.
    """
    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    docs = load_all_uploaded_files()
    if docs:
        _ = vector_store.add_documents(docs)
    else:
        print("No documents found to index.")


def save_uploaded_file(file: UploadFile) -> str:
    """
    Save an uploaded file with a unique filename.
    Returns the new unique filename.
    """
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return unique_filename


def download_and_save(url: str) -> str:
    """
    Download content from a URL and save it as a unique file.
    We assume the URL returns text content.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to download file from URL.")
    content_type = response.headers.get("Content-Type", "")
    # Choose extension based on content type.
    if "application/pdf" in content_type:
        ext = ".pdf"
    elif "text" in content_type:
        ext = ".txt"
    else:
        ext = ".txt"
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    if ext == ".pdf":
        # Save binary content
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        # Save text content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    return unique_filename


def is_google_docs_url(url: str) -> bool:
    """
    Check if the URL is from Google Docs.
    """
    pattern = r"https?://docs\.google\.com/document/d/.*"
    return bool(re.match(pattern, url))


def convert_googledoc_to_pdf_url(url: str) -> str:
    """
    Convert a Google Docs URL to a direct PDF download URL.

    There are two main URL formats:
    1. Published docs: .../pub -> .../export?format=pdf
    2. Direct view: .../edit -> .../export?format=pdf
    """
    # Replace /pub with /export?format=pdf
    if "/pub" in url:
        return url.replace("/pub", "/export?format=pdf")

    # Replace /edit with /export?format=pdf
    elif "/edit" in url:
        return url.replace("/edit", "/export?format=pdf")

    # For other formats with document ID, construct export URL
    elif "/d/" in url:
        # Extract document ID
        doc_id_match = re.search(r"/d/([^/]+)", url)
        if doc_id_match:
            doc_id = doc_id_match.group(1)
            return f"https://docs.google.com/document/d/{doc_id}/export?format=pdf"

    # If we can't determine format, just append export format
    return url + "/export?format=pdf"


def download_google_doc_as_pdf(url: str) -> str:
    """
    Download a Google Doc as PDF and save it.
    Returns the unique filename.
    """
    pdf_url = convert_googledoc_to_pdf_url(url)

    try:
        response = requests.get(pdf_url)

        if response.status_code != 200:
            raise HTTPException(status_code=400,
                                detail=f"Failed to download Google Doc as PDF. Status code: {response.status_code}")

        unique_filename = f"{uuid.uuid4().hex}.pdf"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(file_path, "wb") as file:
            file.write(response.content)

        return unique_filename
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading Google Doc: {str(e)}")


# -------------------------------
# Lifespan Handler
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run startup tasks
    await asyncio.to_thread(reindex_vector_store)
    yield
    # Optional: Add shutdown logic here if needed


# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(
    title="Document QA and Upload Service",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# -------------------------------
# Pipeline Functions for QA
# -------------------------------
def analyze_query(question: str) -> Search:
    analysis_prompt = (
        "Extract the search parameters from the following question. "
        "Return a valid JSON object with two keys: 'query' and 'section'. "
        "The 'query' key should contain the core query string, and the 'section' key should be one of 'beginning', 'middle', or 'end'.\n\n"
        f"Question: {question}\n\n"
        "Output (in JSON):"
    )

    raw_output = llm.invoke(analysis_prompt)
    raw_text = raw_output.content if hasattr(raw_output, "content") else str(raw_output)
    query = structured_parser.parse(raw_text)
    return query


def retrieve(query: Search) -> List[Document]:
    retrieved_docs = vector_store.similarity_search(
        query.query,
        # For demonstration, we ignore a "section" filter here.
        filter=None,
    )
    return retrieved_docs


def generate(question: str, context: List[Document]) -> str:
    docs_content = "\n\n".join(doc.page_content for doc in context)
    prompt_output = prompt.invoke({"question": question, "context": docs_content})
    if hasattr(prompt_output, "content"):
        prompt_text = prompt_output.content
    elif isinstance(prompt_output, list) and prompt_output and hasattr(prompt_output[0], "content"):
        prompt_text = prompt_output[0].content
    else:
        prompt_text = str(prompt_output)
    prompt_text = str(prompt_text)
    response = llm.invoke(prompt_text)
    return response.content if hasattr(response, "content") else str(response)




# -------------------------------
# HealthCheck
# -------------------------------

@app.get("/")
async def health_check():
    """
    Health check endpoint that returns the status of the API.
    """
    return {
        "status": "healthy",
        "service": "Document QA and Upload Service",
        "version": "1.0.0",  # You can add version information
    }

# -------------------------------
# Endpoints for Chat UI
# -------------------------------
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        query = analyze_query(request.question)
        docs_context = retrieve(query)
        answer = generate(request.question, docs_context)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Endpoints for File Upload & CRUD (Upload UI)
# -------------------------------
@app.post("/upload", response_model=FileInfo)
async def upload_file(
        url: str = Form(...)
):
    """
    Upload a file by providing a URL.
    Now handles Google Docs links specifically.
    """
    try:
        if is_google_docs_url(url):
            # Process Google Docs URL
            unique_filename = download_google_doc_as_pdf(url)
            original_filename = "google_doc.pdf"
        else:
            # Process as a regular URL
            unique_filename = download_and_save(url)
            original_filename = url

        # Re-index vector store with new file content
        reindex_vector_store()
        return FileInfo(file_id=unique_filename, original_filename=original_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during upload: {str(e)}")


@app.get("/files", response_model=List[FileInfo])
async def list_files():
    """
    List all uploaded files.
    """
    files = []
    for fname in os.listdir(UPLOAD_DIR):
        files.append(FileInfo(file_id=fname, original_filename=fname))
    return files


@app.get("/files/{file_id}")
async def get_file(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="text/plain", filename=file_id)


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    os.remove(file_path)
    # Re-index vector store after deletion.
    reindex_vector_store()
    return {"detail": "File deleted successfully."}


@app.put("/files/{file_id}", response_model=FileInfo)
async def replace_file(
        file_id: str,
        file: UploadFile = File(...)
):
    """
    Replace the contents of an existing file.
    """
    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    # Overwrite the existing file.
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    reindex_vector_store()
    return FileInfo(file_id=file_id, original_filename=file.filename)


# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    import uvicorn

    # Verify HTML file paths
    index_path = os.path.join(STATIC_DIR, "index.html")
    chatbot_path = os.path.join(STATIC_DIR, "chatbot.html")

    if not os.path.exists(index_path):
        print(f"Warning: index.html not found at {index_path}")
    else:
        print(f"Found index.html at: {os.path.abspath(index_path)}")

    if not os.path.exists(chatbot_path):
        print(f"Warning: chatbot.html not found at {chatbot_path}")
    else:
        print(f"Found chatbot.html at: {os.path.abspath(chatbot_path)}")

    uvicorn.run(app, host="localhost", port=8000)