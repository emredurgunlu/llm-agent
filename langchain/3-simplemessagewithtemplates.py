"""
Introduces ChatPromptTemplate: fill {language} and {text} at runtime,
then run prompt | model | parser as a reusable translation chain.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# Load API keys from .env into environment variables
load_dotenv()


# Chat model via OpenRouter (OpenAI-compatible API)
model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# Template with placeholders filled when you call chain.invoke({...})
system_prompt = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{text}")
])
 

# Extracts .content from AIMessage -> str
parser = StrOutputParser()

# Full LCEL pipeline: build messages -> call LLM -> return plain text
chain = prompt_template | model | parser

if __name__ == '__main__':
    # Pass variables matching the template placeholders
    print(chain.invoke({"language": "English", "text": "Merhaba"}))
