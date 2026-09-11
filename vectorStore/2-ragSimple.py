"""
Simple RAG over hardcoded Documents: retrieve top-k chunks from Chroma,
stuff them into a strict prompt, then let the LLM answer from context only.
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_openai import ChatOpenAI

# Load API keys from .env into environment variables
load_dotenv()

# Embedding model for indexing + query similarity search
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",  # alternative: liquid/lfm-2.5-embedding-350m:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,  # usually not needed with official OpenAI embeddings
    model_kwargs={"encoding_format": "float"},  # usually not needed with official OpenAI embeddings
)

# Sample knowledge base used as RAG context source
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Index documents into Chroma
vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

# Retriever: wrap similarity_search and always return top-1 hit
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

#print(retriever.batch(["cat", "shark"]))



# Generator LLM (answers using retrieved context)
llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# Strict prompt: answer only from {context}, keep it short
message = """
Use ONLY the Context below. Do not add outside knowledge, explanations, or assumptions.
If the answer is not in the Context, say "I don't know based on the provided context."
Quote or lightly restate the Context; keep the answer short.

Question: {question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

# Convert retrieved Document list into a single string for {context}
def format_docs(docs: List[Document]) -> str:
    return "\n".join(doc.page_content for doc in docs)

# RAG chain: retrieve -> format -> prompt -> llm
# RunnablePassthrough passes the raw question into {question}
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

if __name__ == "__main__":
    response = rag_chain.invoke("tell me about cats")
    print(response.content)
