"""
Exposes the translation chain as an HTTP API with FastAPI + LangServe.
Run this file, then call POST /chain/invoke (or use the /docs UI).
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes
import os

# Load API keys from .env into environment variables
load_dotenv()


# Chat model via OpenRouter (OpenAI-compatible API)
model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# Same prompt template as 3-simplemessagewithtemplates
system_prompt = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{text}")
])
 

# Extracts .content from AIMessage -> str
parser = StrOutputParser()

# LCEL pipeline that will be served over HTTP
chain = prompt_template | model | parser

# FastAPI app + LangServe routes around the chain
app = FastAPI()
add_routes(app, chain, path="/chain")

if __name__ == '__main__':
    import uvicorn
    # Start local server: http://0.0.0.0:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)
