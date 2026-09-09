from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free", # Alternatif: liquid/lfm-2.5-embedding-350m:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False, # Resmi OpenAI embedding’de genelde gerekmez.
    model_kwargs={"encoding_format": "float"}, # Resmi OpenAI embedding’de genelde gerekmez.
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()

#  rag prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StringOutputParser()
)



if __name__ == "__main__":
    for chunk in rag_chain.stream("what is maximum inner product search?")