
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings_cache = None


def get_embeddings():
    global _embeddings_cache

    if _embeddings_cache is None:
        print("RAG: Initializing embeddings...", flush=True)

        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("RAG: Embeddings initialized!", flush=True)

        print("RAG: Testing embedding generation...", flush=True)
        test = _embeddings_cache.embed_query("MeetIQ test")
        print(f"RAG: Test embedding size: {len(test)}", flush=True)

    return _embeddings_cache


def build_vector_store(transcript: str):
    print("RAG: Building vector store...", flush=True)

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)
    print(f"RAG: Created {len(chunks)} chunks.", flush=True)

    if not chunks:
        raise ValueError("No text chunks were created.")

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    print("RAG: Generating embeddings...", flush=True)

    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )

    print("RAG: Vector store built successfully.", flush=True)

    return vector_store


def get_retriever(vector_store, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
