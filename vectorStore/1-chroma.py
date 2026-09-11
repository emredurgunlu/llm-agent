"""
Vector store basics: embed short Documents into Chroma and run
similarity search (retrieval only — no LLM generation / not full RAG).
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# Load API keys from .env into environment variables
load_dotenv()

# Embedding model turns text into vectors used for similarity search
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",  # alternative: liquid/lfm-2.5-embedding-350m:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,  # usually not needed with official OpenAI embeddings
    model_kwargs={"encoding_format": "float"},  # usually not needed with official OpenAI embeddings
)

# Sample knowledge base: page_content = text, metadata = source tag
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

# Embed all documents and store vectors in an in-memory Chroma collection
vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

if __name__ == "__main__":
    # print(vectorstore.similarity_search_with_score("tell me about cats"))
    # print(vectorstore.similarity_search("tell me about cats", k=1), )  # k=1: select top result

    # Embed the query, then search by that vector (returns docs + relevance scores)
    query_embedding = embeddings.embed_query("dog")
    print(vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding))
