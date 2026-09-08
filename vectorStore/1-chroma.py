from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free", # Alternatif: liquid/lfm-2.5-embedding-350m:free
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False, # Resmi OpenAI embedding’de genelde gerekmez.
    model_kwargs={"encoding_format": "float"}, # Resmi OpenAI embedding’de genelde gerekmez.
)


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

vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

if __name__ == "__main__":
    # print(vectorstore.similarity_search_with_score("tell me about cats")) 
    # print(vectorstore.similarity_search("tell me about cats", k=1), ) # k=1: select top result

    query_embedding = embeddings.embed_query("dog")
    print(vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding))