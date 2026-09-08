from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()


model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

system_prompt = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{text}")
])
 

parser = StrOutputParser()

chain = prompt_template | model | parser

if __name__ == '__main__':

    print(chain.invoke({"language": "English", "text": "Merhaba"}))