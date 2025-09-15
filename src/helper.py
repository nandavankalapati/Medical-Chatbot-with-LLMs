from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract data from pdf files

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal(docs: List[Document]) -> List[Document]:
    minimal_doc: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_doc.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_doc

def text_split(minimal_doc):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts=text_splitter.split_documents(minimal_doc)
    return texts

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embedding