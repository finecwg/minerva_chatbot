__author__ = 'finecwg'

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain import hub
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, PyPDFLoader, WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from groq import Groq
from langchain_groq import ChatGroq

import os
import pprint
import sys
from dotenv import load_dotenv, find_dotenv
sys.stdout.flush()
os.chdir("/home/user/data/01_ByMember/wgchoi/minerva_chatbot")

_ = load_dotenv(find_dotenv())

#*----dotenv----*
llamaparse_api_key = os.environ['LLAMA_CLOUD_API_KEY']  #* https://cloud.llamaindex.ai/api-key
groq_api_key = os.environ['GROQ_API_KEY']  #* https://console.groq.com/docs/quickstart
tavily_api_key = os.environ['TAVILY_API_KEY']
print(llamaparse_api_key)

from chatting import MinervaQA
minerva_qa = MinervaQA()

query = "Can I bring alcohol to the res hall?"

response = minerva_qa.answer_query(query)