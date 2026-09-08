from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory  # ← ikisi de buradan
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",  # ✅ Mevcut ücretsiz modellerden otomatik seçer
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
    ]
)

chain = prompt | model
config = {"configurable": {"session_id": "123"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

if __name__ == "__main__":
    while True:
        user_input = input(">")
        for r in with_message_history.stream(
            [HumanMessage(content=user_input)],
            config=config,
        ):
            print(r.content, end=" | ")