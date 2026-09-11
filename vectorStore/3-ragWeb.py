"""
Web RAG pipeline:
Load blog HTML -> split into chunks -> embed into Chroma ->
retrieve nearest chunks -> generate an answer with the LLM.
"""

import os
import sys

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from .env into environment variables
load_dotenv()

# Identify this client to remote sites (some block empty User-Agent)
USER_AGENT = "llm-agent-rag-web/1.0"

# Chat model that generates the final answer (OpenRouter free route)
llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,  # 0 = more deterministic / less hallucination
)

# Embedding model that turns text into vectors for search
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",  # alternative: liquid/lfm-2.5-embedding-350m:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,  # send raw text to OpenRouter
    model_kwargs={"encoding_format": "float"},  # request float vectors instead of base64
)

# --- 1) LOAD: download the page and keep only blog content sections ---
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30).text
soup = BeautifulSoup(html, "html.parser")  # parse HTML into a searchable tree
parts = []
for class_name in ("post-title", "post-header", "post-content"):
    for el in soup.select(f".{class_name}"):
        text = el.get_text("\n", strip=True)
        if text:
            parts.append(text)
# LangChain Document: page_content = text, metadata = source info
docs = [Document(page_content="\n\n".join(parts), metadata={"source": url})]

# --- 2) SPLIT: break long text into smaller chunks (fits LLM context better) ---
# chunk_overlap softens cuts mid-sentence by overlapping neighbor chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3) EMBED + STORE: each chunk is vectorized and written to Chroma ---
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# --- 4) RETRIEVE: interface that returns chunks nearest to a query ---
retriever = vectorstore.as_retriever()

# --- 5) PROMPT: tell the model to answer using retrieved context only ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {context}\nAnswer:",
        )
    ]
)

# Convert Document list into one string for the prompt {context} field
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 6) RAG CHAIN: question -> retrieve -> fill prompt -> llm -> plain text ---
# RunnablePassthrough forwards the incoming question into {question}
# | chains steps together (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()  # return str instead of AIMessage
)



if __name__ == "__main__":
    # Avoid Windows console Unicode encoding errors
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # stream: print answer chunks as they arrive
    for chunk in rag_chain.stream("what is maximum inner product search?"):
        print(chunk, end="", flush=True)  # flush=True: write immediately, no buffer wait
