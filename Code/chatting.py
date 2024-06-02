__author__ = 'finecwg'


from langchain.text_splitter import RecursiveCharacterTextSplitter
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

from groq import Groq
from langchain_groq import ChatGroq

import os
import pprint
import sys
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

#*----dotenv----*
llamaparse_api_key = os.environ['LLAMA_CLOUD_API_KEY']  #* https://cloud.llamaindex.ai/api-key
groq_api_key = os.environ['GROQ_API_KEY']  #* https://console.groq.com/docs/quickstart
tavily_api_key = os.environ['TAVILY_API_KEY']

#*----vectorstore----*
vectorstore_minerva_edu = Chroma(
    embedding_function=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"),
    persist_directory="Data/test-db-240529-cms-and-static",
    collection_name="rag"
)

vectorstore_ChatExport = Chroma(
    embedding_function=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"),
    persist_directory="Data/test-db-240530-ChatExport_20240530_1329_M28Official",
    collection_name="rag"
)

vectorstore_Policies_Guidelines = Chroma(
    embedding_function=FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5"),
    persist_directory="Data/test-db-240530-Policies_Guidelines",
    collection_name="rag"
)

retriever_minerva_edu = vectorstore_minerva_edu.as_retriever(search_kwargs={'k': 3})
retriever_ChatExport = vectorstore_ChatExport.as_retriever(search_kwargs={'k': 3})
retriever_Policies_Guidelines = vectorstore_Policies_Guidelines.as_retriever(search_kwargs={'k': 3})

#*----llm----*
llm = ChatGroq(temperature=0.2,
               #format="json",
               model_name="llama3-8b-8192",
               api_key=os.getenv('GROQ_API_KEY'))

# formating docs for RAG
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#*----setting chains----*

#*question_router
prompt_question_router = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search. \n
    Use the vectorstore for questions on Minerva University. \n
    You do not need to be stringent with the keywords in the question related to these topics. \n
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
    Return the a JSON with a single key 'datasource' and no premable or explaination. \n
    Question to route: {question}""",
    input_variables=["question"],
)

question_router = prompt_question_router | llm | StrOutputParser()

#*retrieval_grader
prompt_retrieval_grader = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt_retrieval_grader | llm | StrOutputParser()

#*rag_chain
prompt_rag_chain = PromptTemplate(
    template="""
    <CONTEXT>
    You are an AI assistant who have to help upcoming Minerva University undergraduate freshmen by giving helpful information about Minerva University (which is not an online university).

    Use the following pieces of information to answer the student's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    PLEASE DO NOT PROVIDE THE WRONG ANSWER.

    Context: {context}

    </CONTEXT>

    <OBJECTIVE>
    Provide a detailed and helpful answer to the following question for Minerva University students, who aims to get useful information about the university.

    Question: This is my question about Minerva University. {question}

    Only return the helpful answer below and nothing else.
    Also, please add that your response is based on the user provided information, and it is essential to consult with Minerva University staffs.
    Finally, you must provide the VALID email address of a Minerva University staff member who can be contacted regarding the student's question. If you don't know the valid email address, encourage them to contact [info@minerva.edu](mailto:info@minerva.edu).
    PLEASE do not provide email address that is not included in the context I gave you.
    
    </OBJECTIVE>

    <STYLE>
    This interaction concludes with your one reply, as a markdown format.
    Provide a detailed answer so user don't need to search outside to understand the answer.
    Follow the writing style of desriptive and kind explanation.
    </STYLE>

    <TONE>
    Descriptive
    </TONE>

    <AUDIENCE>
    Audiences are upcoming Minerva University undergraduate freshmans who are curious and need detailed information about Minerva University.
    </AUDIENCE>

    <RESPONSE>
    The descriptive reply, as a markdown format.
    </RESPONSE>
    """,
    input_variables=["question", "document"]
)
#prompt_rag_chain = hub.pull("rlm/rag-prompt")

rag_chain = prompt_rag_chain | llm | StrOutputParser()

#*hauucination_grader
prompt_hallucination_grader = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
     Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt_hallucination_grader | llm | StrOutputParser()

#*answer_grader
prompt_answer_garder = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt_answer_garder | llm | JsonOutputParser()

#*question_rewriter
prompt_question_rewriter = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the initial and formulate an improved question. \n
    Here is the initial question: \n\n {question}. \n 
    Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = prompt_question_rewriter | llm | StrOutputParser()

#*web search tool
web_search_tool = TavilySearchResults(tavily_api_key=tavily_api_key)


###*----LANGGRAPH----*###
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question : str
    generation : str
    documents : List[str]

def retrieve_minerva_edu(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    #st.write("---RETRIEVE---")
    pprint.pprint("---RETRIEVE_MINERVA_EDU---")
    question = state["question"]
    # Retrieval
    documents = retriever_minerva_edu.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def retrieve_ChatExport(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    #st.write("---RETRIEVE---")
    pprint.pprint("---RETRIEVE_CHATEXPORT---")
    question = state["question"]

    # Retrieval
    prior_documents = state["documents"]
    temp_documents = retriever_ChatExport.invoke(question)

    # Ensure both prior_documents and temp_documents are lists of Document objects
    if isinstance(prior_documents, str):
        prior_documents = [Document(page_content=prior_documents)]

    documents = format_docs(prior_documents + temp_documents)
    return {"documents": documents, "question": question}

def retrieve_Policies_Guidelines(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    #st.write("---RETRIEVE---")
    pprint.pprint("---RETRIEVE_POLICIES_GUIDELINES---")
    question = state["question"]

    # Retrieval
    prior_documents = state["documents"]
    temp_documents = retriever_Policies_Guidelines.invoke(question)

    # Ensure both prior_documents and temp_documents are lists of Document objects
    if isinstance(prior_documents, str):
        prior_documents = [Document(page_content=prior_documents)]
    
    documents = format_docs(prior_documents + temp_documents)
    return {"documents": documents, "question": question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    #st.write("---WEB SEARCH---")
    pprint.pprint("----WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    
    # Ensure docs is a list of dictionaries
    if isinstance(docs, str):
        docs = [Document(page_content=docs)]
    else:
        docs = [Document(page_content=d["content"]) for d in docs]

    
    prior_documents = state["documents"]

    # Ensure prior_documents is a list of Document objects
    if isinstance(prior_documents, str):
        prior_documents = [Document(page_content=prior_documents)]
    
    documents = format_docs(prior_documents + docs)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    #st.write("---GENERATE---")
    pprint.pprint("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    

    pprint.pprint("User's question:")
    pprint.pprint(question)
    pprint.pprint("Retrieved documents: ---")
    pprint.pprint(documents)
        
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    #st.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    pprint.pprint("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
        
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade == "yes":
            #st.write("---GRADE: DOCUMENT RELEVANT---")
            pprint.pprint("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            #st.write("---GRADE: DOCUMENT NOT RELEVANT---")
            pprint.pprint("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    #st.write("---TRANSFORM QUERY---")
    pprint.pprint("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print("documents in transform_query")
    print(documents)
    print("question in transform_query")
    print(better_question)
    return {"documents": documents, "question": better_question}

def start(state):
    pprint.pprint("----START---")
    return state


### Edges ###

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    #st.write("---ROUTE QUESTION---")
    pprint.pprint("----ROUTE QUESTION---")
    question = state["question"]
    #st.write(question)
    source = question_router.invoke({"question": question})
    #st.write(source)
    pprint.pprint(source)
    #st.write(source['datasource'])
    pprint.pprint(source['datasource'])
    if source['datasource'] == 'web_search':
        #st.write("---ROUTE QUESTION TO WEB SEARCH---")
        pprint.pprint("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source['datasource'] == 'vectorstore':
        #st.write("---ROUTE QUESTION TO RAG---")
        pprint.pprint("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    #st.write("---ASSESS GRADED DOCUMENTS---")
    pprint.pprint("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        #st.write("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        pprint.pprint("---DECISION:ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        #st.write("---DECISION: GENERATE---")
        pprint.pprint("---GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    #st.write("---CHECK HALLUCINATIONS---")
    pprint.pprint("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        #st.write("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        pprint.pprint("--DECISION:GENERSTION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        #st.write("---GRADE GENERATION vs QUESTION---")
        pprint.pprint("--GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            #st.write("---DECISION: GENERATION ADDRESSES QUESTION---")
            pprint.pprint("--DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            #st.write("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            pprint.pprint("--DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        #st.write("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        pprint.pprint("--DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


#*----workflow with langgraph----*

workflow = StateGraph(GraphState)

# Define the nodesAttributeError: 'StateGraph' object has no attribute 'set_conditional_entry_point'
workflow.add_node("start",start) # start
workflow.add_node("web_search", web_search) # web search
workflow.add_node("retrieve_minerva_edu", retrieve_minerva_edu)
workflow.add_node("retrieve_ChatExport", retrieve_ChatExport)
workflow.add_node("retrieve_Policies_Guidelines", retrieve_Policies_Guidelines) # retrieve
#workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generate
workflow.add_node("transform_query", transform_query) # transform_query

#*----Build graph----*

workflow.set_entry_point("start")
workflow.add_edge("start", "transform_query")
workflow.add_edge("transform_query", "retrieve_minerva_edu")
workflow.add_edge("retrieve_minerva_edu", "retrieve_ChatExport")
workflow.add_edge("retrieve_ChatExport", "retrieve_Policies_Guidelines")
workflow.add_edge("retrieve_Policies_Guidelines", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

#*----compile----*

class MinervaQA:
    def __init__(self):
        self.app = workflow.compile()
    
    def answer_query(self, query):
        pprint.pprint(query)
        inputs = {"question" : query}
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                #st.write(f"Node '{key}':")
                # Optional: print full state at each node

                pprint.pprint(key, indent=2, width=80, depth=None)
            print("\n---\n")   
        #Final generation
        #st.write(value["generation"])
        pprint.pprint(value["generation"], indent=2, width=80, depth=None)
        sys.stdout.flush()

        return value["generation"]

''' #*testing the response
minerva_qa = MinervaQA()
query = "Explain me about Minerva"
response = minerva_qa.answer_query(query)
pprint.pprint(response)
'''

'''
app = workflow.compile()
user_input="Explain about Minerva University"
inputs = {"question" : user_input}

for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        #st.write(f"Node '{key}':")
        # Optional: print full state at each node

        pprint.pprint(key, indent=2, width=80, depth=None)
    print("\n---\n")

# Final generation
#st.write(value["generation"])
pprint.pprint(value["generation"], indent=2, width=80, depth=None)
'''

