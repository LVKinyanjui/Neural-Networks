import os
from dotenv import load_dotenv, find_dotenv
import warnings
import datetime

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

# Read local .env file
_ = load_dotenv(find_dotenv())

# Ignore warnings
warnings.filterwarnings("ignore")

# Account for deprecation of LLM model
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)

if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# Create ChatOpenAI instance
llm = ChatOpenAI(temperature=0, model=llm_model)

# Load built-in tools
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Example question to the agent
question = f"What is the square root of 2"
agent(question)

# Wikipedia example
question = "Tom M. Mitchell is an American computer scientist and the Founders University Professor at Carnegie Mellon University (CMU) what book did he write?"
result = agent(question)

# Create Python agent
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

# Example Python agent usage
customer_list = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"]
]

agent.run(f"Sort these customers by last name and then first name and print the output: {customer_list}")

# View detailed outputs of the chains
import langchain
langchain.debug = True
agent.run(f"Sort these customers by last name and then first name and print the output: {customer_list}")
langchain.debug = False

# Define your own tool
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """Returns today's date, use this for any questions related to knowing today's date.
    The input should always be an empty string, and this function will always return today's
    date - any date mathematics should occur outside this function."""
    return str(date.today())

# Initialize agent with custom tool
agent = initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Example usage of custom tool
try:
    result = agent("whats the date today?")
except:
    print("exception on external access")
