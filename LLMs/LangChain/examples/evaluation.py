import os
from dotenv import load_dotenv, find_dotenv
import datetime

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

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

# Create QandA application
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature=0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)

# Coming up with test datapoints
data[10]
data[11]

# Hard-coded examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# LLM-Generated examples
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

# the warning below can be safely ignored
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

new_examples[0]
data[0]

# Combine examples
examples += new_examples

qa.run(examples[0]["query"])

# Manual Evaluation
import langchain
langchain.debug = True

qa.run(examples[0]["query"])

# Turn off the debug mode
langchain.debug = False

# LLM assisted evaluation
predictions = qa.apply(examples)

llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()

graded_outputs[0]

# LangChain evaluation platform
# The LangChain evaluation platform, LangChain Plus, can be accessed here https://www.langchain.plus/.
# Use the invite code `lang_learners_2023`
# Reminder: Download your notebook to your local computer to save your work.
