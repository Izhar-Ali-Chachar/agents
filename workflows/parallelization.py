from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import os

# Load .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize model
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Define agent state
class AgentState(TypedDict):
    topic: str
    characters: str
    settings: str
    promises: str
    combines: str

# Define nodes
def char_node(state: AgentState) -> AgentState:
    prompt = f"Create two characters and a short story idea on topic: {state['topic']}."
    response = llm.invoke(prompt)
    return {"characters": response.content}

def setting_node(state: AgentState) -> AgentState:
    prompt = f"Write a short setting description on topic: {state['topic']}."
    response = llm.invoke(prompt)
    return {"settings": response.content}

def promise_node(state: AgentState) -> AgentState:
    prompt = f"Suggest a few story promises or dramatic tensions based on topic: {state['topic']}."
    response = llm.invoke(prompt)
    return {"promises": response.content}

def combine_node(state: AgentState) -> AgentState:
    prompt = (
        f"Write a detailed story combining the following:\n"
        f"1. Characters: {state['characters']}\n"
        f"2. Settings: {state['settings']}\n"
        f"3. Promises: {state['promises']}"
    )
    response = llm.invoke(prompt)
    return {"combines": response.content}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("char", char_node)
graph.add_node("setting", setting_node)
graph.add_node("promise", promise_node)
graph.add_node("combine", combine_node)

graph.add_edge(START, "char")
graph.add_edge(START, "setting")
graph.add_edge(START, "promise")
graph.add_edge("char", "combine")
graph.add_edge("setting", "combine")
graph.add_edge("promise", "combine")
graph.add_edge("combine", END)

# Compile the agent
agent = graph.compile()


from IPython.display import Image, display

# Generate graph image
graph_image = agent.get_graph().draw_mermaid_png()
display(Image(graph_image))
