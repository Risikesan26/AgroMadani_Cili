import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

ARTICLES_DIR = "Articles"
VECTOR_STORE_DIR = "faiss_index"

def load_documents():
    pdf_files = glob.glob(os.path.join(ARTICLES_DIR, "*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in '{ARTICLES_DIR}' directory.")
    
    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    return documents

def build_vector_store():
    documents = load_documents()
    if not documents:
        raise ValueError("No documents were loaded. Cannot build vector store.")
        
    print(f"Loaded {len(documents)} pages. Splitting text...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks. Generating embeddings and building vector store...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    print("Vector store built successfully.")
    return vector_store

def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(VECTOR_STORE_DIR):
        return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Vector store not found. Building it for the first time...")
        return build_vector_store()

def setup_qa_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    system_prompt = (
        "You are an expert assistant on chili plant diseases. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise and relevant.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def answer_question(query: str):
    vector_store = get_vector_store()
    rag_chain = setup_qa_chain(vector_store)
    response = rag_chain.invoke({"input": query})
    return response["answer"]
