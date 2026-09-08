from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
model = ChatOpenAI(
    model="openrouter/free",  # ✅ Mevcut ücretsiz modellerden otomatik seçer
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

if __name__ == '__main__':
    response = model.invoke(messages)
    print(response)
    # print(response.content)