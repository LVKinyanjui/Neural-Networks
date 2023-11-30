import os
from dotenv import load_dotenv, find_dotenv
import datetime

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from IPython.display import display, Markdown

# Read local .env file
_ = load_dotenv(find_dotenv())

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Account for deprecation of LLM model
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)

if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# Set up Q&A over Documents
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# Set up Vectorstore Index
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# Query and display results in Markdown
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query)
display(Markdown(response))

# Step By Step

# Load documents
docs = loader.load()
docs[0]

# Embed queries
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi, my name is Harrison")
print(len(embed))
print(embed[:5])

# Create DocArrayInMemorySearch
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Perform similarity search
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
len(docs)
docs[0]

# Convert to retriever
retriever = db.as_retriever()

# Set up ChatOpenAI for Q&A
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Concatenate document pages for Q&A
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

# Perform Q&A using concatenated documents
response = llm.call_as_llm(f"{qdocs} Question: {query}")
display(Markdown(response))

# Set up RetrievalQA for Q&A
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

# Run Q&A and display response
response = qa_stuff.run(query)
display(Markdown(response))

# Additional Query using Vectorstore Index with embeddings
response = index.query(query, llm=llm)

# Create Vectorstore Index with embeddings
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

# Reminder: Download your notebook to your local computer to save your work.
