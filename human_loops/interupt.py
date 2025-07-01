from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import os
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up memory
memory = MemorySaver()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Define a tool
@tool
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b

tools = [add]

# LLM with tool support
llm_with_tools = llm.bind_tools(tools)

# Define graph state
class State(TypedDict):
    messages: list

# Human feedback node (pause for user input)
def human_feedback(state: State) -> State:
    return state  # This just pauses for user input

# Assistant node (LLM response)
def assistant(state: State) -> State:
    messages = state["messages"]
    new_message = llm_with_tools.invoke(messages)
    return {"messages": messages + [new_message]}

# Build the graph
graph = StateGraph(State)

graph.add_node("humanfed", human_feedback)
graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "humanfed")
graph.add_edge("humanfed", "assistant")

# Fix: typo was 'assistatn'
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "humanfed")

# Compile agent with memory and interruption
agent = graph.compile(interrupt_before=["humanfed"], checkpointer=memory)

# Configuration (e.g., thread ID for memory checkpointing)
config = {"configurable": {"thread_id": "4"}}

# Initial input message
initial_state = {
    "messages": [HumanMessage(content="add 2 and 4")]
}

# Run the graph with streaming output
for step in agent.stream(initial_state, config=config, stream_mode="values"):
    step["messages"][-1].pretty_print()


