"""
Same chat-with-memory setup as 5-messagingHistory, but streams
token chunks with .stream() instead of waiting for a full invoke.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory  # both come from here
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# Load API keys from .env into environment variables
load_dotenv()

# Chat model via OpenRouter (OpenAI-compatible API)
model = ChatOpenAI(
    model="openrouter/free",  # auto-picks from available free models
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# session_id -> chat history store (in-memory, lost on restart)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Create a new history bucket the first time a session is seen
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Prompt injects past messages via MessagesPlaceholder("history")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
    ]
)

chain = prompt | model
# Ties this run to session "123" so history is reused
config = {"configurable": {"session_id": "123"}}
# Wraps the chain: loads history before call, saves after
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

if __name__ == "__main__":
    # REPL with streaming: print each chunk as it arrives
    while True:
        user_input = input(">")
        for r in with_message_history.stream(
            [HumanMessage(content=user_input)],
            config=config,
        ):
            print(r.content, end=" | ")
