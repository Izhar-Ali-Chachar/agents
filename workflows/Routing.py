from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Literal
import os
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Pydantic model for routing
class Routing(BaseModel):
    step: Literal['poem', 'joke', 'story'] = Field(description="routing step")

# Model with structured output
router = llm.with_structured_output(Routing)

# Agent state definition
class AgentState(TypedDict):
    input: str
    decision: str
    output: str

# Story node
def llm_call_1(state: AgentState) -> AgentState:
    response = llm.invoke(state['input'])
    return {'input': state['input'], 'decision': state['decision'], 'output': response.content}

# Joke node
def llm_call_2(state: AgentState) -> AgentState:
    response = llm.invoke(state['input'])
    return {'input': state['input'], 'decision': state['decision'], 'output': response.content}

# Poem node
def llm_call_3(state: AgentState) -> AgentState:
    response = llm.invoke(state['input'])
    return {'input': state['input'], 'decision': state['decision'], 'output': response.content}

# Routing node
def llm_call_router(state: AgentState) -> AgentState:
    decision = router.invoke([
        SystemMessage(content="Route the input to story, joke, or poem based on user request."),
        HumanMessage(content=state['input'])
    ])
    return {'input': state['input'], 'decision': decision.step, 'output': ''}

# Decision logic
def route_decision(state: AgentState):
    if state['decision'] == 'poem':
        return 'llm_call_3'
    elif state['decision'] == 'joke':
        return 'llm_call_2'
    elif state['decision'] == 'story':
        return 'llm_call_1'

# Build graph
graph = StateGraph(AgentState)

graph.add_node('llm_call_router', llm_call_router)
graph.add_node('llm_call_1', llm_call_1)
graph.add_node('llm_call_2', llm_call_2)
graph.add_node('llm_call_3', llm_call_3)

graph.add_edge(START, 'llm_call_router')
graph.add_conditional_edges('llm_call_router', route_decision, {
    'llm_call_1': 'llm_call_1',
    'llm_call_2': 'llm_call_2',
    'llm_call_3': 'llm_call_3'
})
graph.add_edge('llm_call_1', END)
graph.add_edge('llm_call_2', END)
graph.add_edge('llm_call_3', END)

# Compile the agent
agent = graph.compile()

# Invoke the agent with an input
result = agent.invoke({'input': 'Write a short one-line joke', 'decision': '', 'output': ''})
print(result['output'])
