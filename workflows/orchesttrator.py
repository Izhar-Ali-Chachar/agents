from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os
import operator
import Send
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Section(BaseModel):
    name: str = Field(description="this will generate the name of section")
    description: str = Field(description="brief overview of main topics and concepts of section")

class Sections(BaseModel):
    sections: list[Section]= Field(description="sections of report")

llm_call = llm.with_structured_output(Sections)

class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: list[Section]
    completed_sections: Annotated[list, operator.add]


def orchestrator(state: State):
    """orchestrator that generate plan for report"""

    response_generator = llm_call.invoke(
        [
            SystemMessage(content="generate plan for report"),
            HumanMessage(content=state['topic'])
        ]
    )

    return {'sections': response_generator.sections}

def llm_call_1(state: WorkerState):
    """worker writes the section for report"""

    responser_generator = llm_call.invoke(
        [
            SystemMessage(content="write the report section following the provide name and description"),
            HumanMessage(content=f'here is section name {state['section'].name} and description {state['section'].description}')
        ]
    )

    return {'comleted_sections': responser_generator.content}

def assign_worker(state: State):
   """assign a worker to each section in the plan"""

   return [Send(llm_call_1, {'section': s}) for s in state['sections']]


def synthesizer(state: State):
    """combine all the sections"""

    completed_sections = state['completed_sections']

    completed_report_section = "\n\n---\n\n".join(completed_sections)

    return {'final_report': completed_report_section}


graph = StateGraph(State)

graph.add_node('llm_call', llm_call_1)
graph.add_node('orchestrator', orchestrator)
graph.add_node('synthesizer', synthesizer)


graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges(
    'orchestrator',
    assign_worker,
    ['llm_call']
)

graph.add_edge('llm_call', 'synthesizer')
graph.add_edge('synthesizer', END)

agent = graph.compile()

print(agent.ivoke('write report on generative ai'))