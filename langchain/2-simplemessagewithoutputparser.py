"""
Same basic chat as 1-simplemessage, but pipes the model through
StrOutputParser so the chain returns plain text instead of AIMessage.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os

# Load API keys from .env into environment variables
load_dotenv()


# Chat model via OpenRouter (OpenAI-compatible API)
model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# Conversation payload: system sets behavior, human is the user question
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

# Extracts .content from AIMessage -> str
parser = StrOutputParser()
# response = model.invoke(messages)
# LCEL chain: model output flows into the parser
chain = model | parser

if __name__ == '__main__':
    # print(parser.invoke(response))
    # invoke runs the whole chain and prints a plain string
    print(chain.invoke(messages))
