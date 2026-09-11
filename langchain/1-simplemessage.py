"""
Minimal LangChain chat example: send System + Human messages to an LLM
and print the raw AIMessage response (no parser, no chain).
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

# Load API keys from .env into environment variables
load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# Chat model via OpenRouter (OpenAI-compatible API)
model = ChatOpenAI(
    model="openrouter/free",  # auto-picks from available free models
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# Conversation payload: system sets behavior, human is the user question
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

if __name__ == '__main__':
    # invoke = one-shot call; returns an AIMessage object
    response = model.invoke(messages)
    print(response)
    # print(response.content)
