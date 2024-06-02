__author__ = 'finecwg'

'''
*making vector DB for RAG
'''

import os
from dotenv import load_dotenv, find_dotenv

os.chdir("/home/user/data/01_ByMember/wgchoi/minerva_chatbot")

_ = load_dotenv(find_dotenv())

llamaparse_api_key = os.environ['LLAMA_CLOUD_API_KEY'] #* https://cloud.llamaindex.ai/api-key
groq_api_key = os.environ['GROQ_API_KEY'] #* https://console.groq.com/docs/quickstart

##### LLAMAPARSE #####
from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#
from groq import Groq
from langchain_groq import ChatGroq
#
import joblib
import os
import argparse
import nest_asyncio  # noqa: E402
nest_asyncio.apply()

def get_file_list(format, *file_dirs):
    if format not in ['pdf', 'html']:
        raise ValueError("Format must be 'pdf' or 'html'")

    file_list = []

    # Function to list files with the specified format in a directory
    def list_files(directory):
        for filename in os.listdir(directory):
            if filename.endswith(f'.{format}'):
                file_list.append(os.path.join(directory, filename))

    # List files from each provided directory
    for directory in file_dirs:
        list_files(directory)

    return file_list




def load_or_parse_data(format = 'html',
                       pkl_data_file = "Data/parsed_data_240529_cms.pkl", 
                       raw_file_path = 'Data/scrapped_html/sitemap-cms',
                       parsing_instruction = "The provided document contains information about Minerva University."):
                       
    data_file = pkl_data_file

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        
        docs = get_file_list(format, raw_file_path) ##TODO
    
        # Perform the parsing step and store the result in llama_parse_documents
        '''
        parsingInstructionUber10k = """The provided document contains information about Minerva University.
        This form provides detailed information about Minerva University and their educational programs, and stories.\
        It contains many tables and figures.
        Try to be precise while answering the questions"""
        '''
        parsingInstructionUber10k = parsing_instruction

        parser = LlamaParse(api_key=llamaparse_api_key,
                            result_type="markdown",
                            parsing_instruction=parsingInstructionUber10k,
                            max_timeout=5000,)
        llama_parse_documents = parser.load_data(docs)


        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data

# Create vector database
def create_vector_database(format = 'html',
                           db_directory = "Data/test-db-240529-cms", 
                           pkl_data_file = "./Data/parsed_data_240529_cms.pkl", 
                           raw_file_path = 'Data/scrapped_html/sitemap-cms',
                           parsing_instruction = "The provided document contains information about Minerva University."):
    """
    *Creates a vector database using document loaders and embeddings.

    *This function loads urls,
    *splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    *and finally persists the embeddings into a Chroma vector database.

    """
    db_directory = db_directory
    os.makedirs(db_directory, exist_ok =True)
    os.chmod(db_directory, 0o755)



    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data(format = format,
                                               pkl_data_file = pkl_data_file, 
                                               raw_file_path = raw_file_path,
                                               parsing_instruction = parsing_instruction)
    
    print(llama_parse_documents[0].text[:300])

    with open(os.path.join(db_directory, "output.md"), 'a') as f:  #TODO
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = os.path.join(db_directory, "output.md") ##TODO
    loader = UnstructuredMarkdownLoader(markdown_path)

   #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    #docs[0]

    # Initialize Embeddings
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory=db_directory,  # Local mode with in-memory storage only
        collection_name="rag"
    )

    #query it
    #query = "what is the agend of Financial Statements for 2022 ?"
    #found_doc = qdrant.similarity_search(query, k=3)
    #print(found_doc[0][:100])
    #print(qdrant.get())

    print('Vector DB created successfully !')
    return vs,embed_model

def main(args):
    create_vector_database(format = args.format, 
                           db_directory = args.db_directory, 
                           pkl_data_file = args.pkl_data_file, 
                           raw_file_path = args.raw_file_path,
                           parsing_instruction = args.parsing_instruction)



def parse_args():
    parser = argparse.ArgumentParser(description="Creating Vector DB")

    parser.add_argument('--format', type=str, choices=['html', 'pdf'], default='html', help="File format: 'html' or 'pdf'")
    parser.add_argument('--db_directory', type=str, required=True, help="Chroma DB directory")
    parser.add_argument('--pkl_data_file', type = str, required = True, help = ".pkl file directory")
    parser.add_argument('--raw_file_path', type = str, required = True, help = "raw file directory (html or pdf)")
    parser.add_argument('--parsing_instruction', type=str, required = True, help = "Instruction for creating Vector DB")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)