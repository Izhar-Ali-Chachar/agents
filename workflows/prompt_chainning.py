from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("GOOGLE_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


class AgentState(TypedDict):
    messages: str
    story: str
    improve_story: str
    final_story: str

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

def generate_story(state: AgentState) -> AgentState:
    response = llm.invoke(f"write one line story promise staet['messages']")
    return {'story': response.content}

def improved_story(state: AgentState) -> AgentState:
    response = llm.invoke(f"enhance the story with detail promises staet['stroy']")
    return {'improve_story': response.content}

def polished_story(state: AgentState) -> AgentState:
    response = llm.invoke(f"add an unexpected twist in story staet['improve_story']")
    return {'improve_story': response.content}

def should_continue(state: AgentState):
    if '?' in state['story'] or '!' in state['story']:
        return 'fail'
    return 'pass'


def make_default_graph():

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node('generate_story', generate_story)
    graph_builder.add_node('improved_story', improved_story)
    graph_builder.add_node('polished_story', polished_story)

    graph_builder.add_edge(START, 'generate_story')
    graph_builder.add_conditional_edges(
        'generate_story',
         should_continue,
         {'fail': 'generate_story', 'pass': 'improved_story'}
        )
    graph_builder.add_edge('generate_story', 'improved_story')
    graph_builder.add_edge('improved_story', 'polished_story')
    graph_builder.add_edge('polished_story', END)

    graph = graph_builder.compile()

    return graph

graph = make_default_graph()
