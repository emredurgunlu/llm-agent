from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()


model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

parser = StrOutputParser()
# response = model.invoke(messages)
chain = model | parser

if __name__ == '__main__':
    # print(parser.invoke(response))
    print(chain.invoke(messages))