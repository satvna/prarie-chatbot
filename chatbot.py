import os
import pinecone

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone

index_name = pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index(index_name)

#----- Create new vector database if one does not exist: ----
try:
  index_info = pinecone.describe_index('example-index')
except:
  index_info = pinecone.create_index("example-index", dimension=768)

#get existing vector count 
print(index)

embeddings = HuggingFaceHubEmbeddings()

# ----- Add info to vector store (if not already loaded) ----
if(index_info.status['ready']):
  docsearch = Pinecone.from_existing_index('example-index', embeddings)
else: 
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

# Sending query
chat_history = []
query = "How many acres of the original 12 million acres of Blackland Prarie remain in true prarie condition?"
result = qa({"question": query, "chat_history": chat_history})
print(result)