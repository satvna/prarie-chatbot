import os
import sys
import pinecone
import prompts

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone

green = "\033[0;32m" #decorative colors for terminal
white = "\033[0;39m"
yellow = "\033[0;33m"

index_name = pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index(index_name)
index_is_new = False

#----- Create new vector database if one does not exist: ----
try:
  index_info = pinecone.describe_index('example-index')
  index_is_new = False
except:
  index_info = pinecone.create_index("example-index", dimension=768)
  index_is_new =  True

#get existing vector count 
print(index)

embeddings = HuggingFaceHubEmbeddings()

# ----- Add info to vector store (if not already loaded) ----
if(index_is_new == False):
  print("Importing index...")
  docsearch = Pinecone.from_existing_index('example-index', embeddings)
else: 
  print("Creating index...")
  # ----- Splitting, embedding, and storing PDFs in Pinecone  -----
  documents = []
  for file in os.listdir('documents'):
    doc_path = './documents/' + file
    loader = PyPDFLoader(doc_path)
    documents.extend(loader.load())

  #split docs into paragraphs of 1000 chars or fewer
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
  split_text = text_splitter.split_documents(documents)

  #embed and store
  docsearch = Pinecone.from_documents(split_text, embeddings, index_name="example-index")


# ----- Setting up HuggingFace model & query -----
repo_id = "google/flan-t5-base"
llm = HuggingFaceHub(huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_KEY"),
    repo_id=repo_id,
    model_kwargs={"temperature":0, "max_length":64})

qa= ConversationalRetrievalChain.from_llm(
      llm,
      retriever= docsearch.as_retriever())

#--- Loop so that user can ask multiple queries ---
chat_history=[]
while True:
  prompt_to_send = ""

  #--- Commands for loop ---
  user_prompt = input(f"{green}Prompt: ")
  if user_prompt == "exit" or user_prompt== "f" or user_prompt == "quit" or user_prompt == "q": 
    print(f'{white}Exiting...\n')
    sys.exit()
  if user_prompt == '' or user_prompt == '!':
    continue       
  if user_prompt.startswith("!"): #To use the variable name of pretyped query, use "!". Example: !query"
    if hasattr(prompts, user_prompt[1:]):
      prompt_to_send = getattr(prompts, str(user_prompt[1:]))
    else: 
      print(f'{white}Error: Specified query does not exist.\n')
      continue
    print(f"{green}Prewritten prompt: " + prompt_to_send)
  else:
    prompt_to_send = user_prompt

  #--- Send query ---    
  result = qa({"question": prompt_to_send, "chat_history": chat_history})
  print(f'{yellow}{result["answer"]}')
  # print(f'{white}{result["chat_history"]}')
  # print(f'{yellow}{chat_history}')


  chat_history.append((prompt_to_send, result["answer"]))

  #sys.exit()