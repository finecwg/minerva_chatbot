__author__ = 'finecwg'

#require: pip install python-telegram-bot 
# pip install -U langgraph 

##### LLAMAPARSE #####
from llama_parse import LlamaParse


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from groq import Groq
from langchain_groq import ChatGroq

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

import joblib
import os


from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

llamaparse_api_key = os.environ['LLAMA_CLOUD_API_KEY'] #* https://cloud.llamaindex.ai/api-key
groq_api_key = os.environ['GROQ_API_KEY'] #* https://console.groq.com/docs/quickstart
tavily_api_key = os.environ['TAVILY_API_KEY']
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']


chat_model = ChatGroq(temperature=0.2,
                      model_name="llama3-8b-8192",
                      api_key=os.getenv('GROQ_API_KEY'),)
vectorstore = Chroma(embedding_function=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"),
                      persist_directory="Data/test-db-240529-cms-and-static",
                      collection_name="rag")
 
retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

custom_prompt_template = """You are an tutor who have to give information about Minerva University to students who are curious about Minerva University.

This interaction concludes with your one reply, as a markdown format.

Use the following pieces of information to answer the student's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Also, please add that your response is based on the user provided information, and it is essential to consult with a qualified veterinarian.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
#
prompt = set_custom_prompt()


qa = RetrievalQA.from_chain_type(llm=chat_model,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})






async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hi! Ask me anything about Minerva University.')

async def handle_message(update: Update, context: CallbackContext) -> None:
    query = update.message.text
    response = qa.invoke({"query": query})
    await update.message.reply_text(response['result'])

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == '__main__':
    main()