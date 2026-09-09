"""
Akış:
Load — Blog HTML’ini çek, sadece yazı kısımlarını al
Split — Uzun metni chunk’lara böl
Embed + Store — Chunk’ları vektöre çevir, Chroma’ya yaz
Retrieve — Soruya en yakın chunk’ları bul
Generate — O chunk’ları prompt’a koy, LLM cevap üretsin
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

# .env içindeki API anahtarlarını ortam değişkenine yükler
load_dotenv()

# Siteler bot isteklerini tanısın diye User-Agent gönderiyoruz
USER_AGENT = "llm-agent-rag-web/1.0"

# Cevabı üretecek chat modeli (OpenRouter ücretsiz rota)
llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,  # 0 = daha deterministik / az "uydurma"
)

# Metni sayısal vektöre çeviren embedding modeli (arama bununla yapılır)
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",  # Alternatif: liquid/lfm-2.5-embedding-350m:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,  # OpenRouter'a ham metin gönder
    model_kwargs={"encoding_format": "float"},  # base64 yerine float vektör al
)

# --- 1) LOAD: web sayfasını indirip sadece blog içeriğini çıkar ---
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30).text
soup = BeautifulSoup(html, "html.parser")  # HTML'i parse eder
parts = []
for class_name in ("post-title", "post-header", "post-content"):
    for el in soup.select(f".{class_name}"):
        text = el.get_text("\n", strip=True)
        if text:
            parts.append(text)
# LangChain Document: page_content = metin, metadata = kaynak bilgisi
docs = [Document(page_content="\n\n".join(parts), metadata={"source": url})]

# --- 2) SPLIT: uzun metni küçük parçalara böler (LLM context penceresi için) ---
# chunk_overlap: komşu parçalar biraz örtüşsün diye; cümle ortasında kopmayı yumuşatır
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3) EMBED + STORE: her chunk vektöre çevrilip Chroma'ya yazılır ---
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# --- 4) RETRIEVE: soruya semantik olarak en yakın chunk'ları getiren arayüz ---
retriever = vectorstore.as_retriever()

# --- 5) PROMPT: modele "sadece context'e bakarak cevap ver" der ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {context}\nAnswer:",
        )
    ]
)

# Document listesini tek string'e çevirir (prompt {context} alanına gider)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 6) RAG CHAIN: soru -> retrieve -> prompt doldur -> llm -> düz metin ---
# RunnablePassthrough: gelen soruyu olduğu gibi "question" alanına iletir
# | operatörü adımları zincirler (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()  # AIMessage yerine sadece string döndür
)



if __name__ == "__main__":
    # Windows konsolunda özel karakter bozulmasın
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # stream: cevabı parça parça basar (beklerken canlı görünür)
    for chunk in rag_chain.stream("what is maximum inner product search?"):
        print(chunk, end="", flush=True)  # flush=True: buffer beklemeden hemen yaz
