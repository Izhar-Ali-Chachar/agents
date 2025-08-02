from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # This is the main embedding model available
)

urls=[
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

doc_splits = text_splitter.split_documents(docs_list)

vectore_store = FAISS.from_documents(
    documents=doc_splits,
    embedding=embedding_model
)

retriever = vectore_store.as_retriever()

retrieverlanggraph = create_retriever_tool(
    retriever,
    "retriever_vector_db_blog",
    "Search and run information about Langgraph"
)

langchain_urls=[
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

docs=[WebBaseLoader(url).load() for url in langchain_urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

doc_splits = text_splitter.split_documents(docs_list)

## Add alll these text to vectordb

vectorstorelangchain=FAISS.from_documents(
    documents=doc_splits,
    embedding=embedding_model
)


retrieverlangchain = vectorstorelangchain.as_retriever()

retrieverlangchain_tool = create_retriever_tool(
    retrieverlangchain,
    "retriever_vector_db_langchain",
    "Search and run information about LangChain"
)

tools = [retrieverlanggraph, retrieverlangchain_tool]


from typing import Annotated, TypedDict, Literal, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class State(TypedDict):
    """
    State for the agent.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def agent(state: State):
    """
    agent fuction that takes a query and returns the relevant document from the vector store.

    it uses the retriever tools to search for relevant information in the vector store.

    input:
        state (messages): State object containing the messages.

    output:
        dict: updated state with the response and append the response to the messages.
    """

    query = state["messages"][0].content

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(query)
    return {"messages": [response]}

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

def grade_document(state: State) -> dict:
    """
    Function to grade the context based on the question. 

    input:
        state (messages): State object containing the messages.

    output:
        dict: {'next': 'rewrite'} if the response is not satisfactory, {'next': 'generate'} if it is.
    """
    
    class grade(BaseModel):
        """
        Model to grade the response.
        """
        binary_score: str = Field(
            ...,
            description="The response to be graded."
        )

    llm_with_structured_output = llm.with_structured_output(grade)

    prompt = PromptTemplate(
        template="Grade the following response based on the question. " \
        "here is the response: {context}. " \
        "here is the question: {query}. " \
        "If the response is satisfactory, return 'generate'. " \
        "If the response is not satisfactory, return 'rewrite'.",
        input_variables=["context", "question"]
    )

    query = state["messages"][0].content
    context = state["messages"][-1].content

    formatted_prompt = prompt.format(context=context, query=query)
    scored_response = llm_with_structured_output.invoke(formatted_prompt)

    if scored_response.binary_score == "generate":
        return {"next": "generate"}
    else:
        return {"next": "rewrite"}


def generate_response(state: State):
    """
    Function to generate a response based on the query and context.
    """
    class Response(BaseModel):
        response: str = Field(
            ...,
            description="The generated response."
        )

    llm_with_structured_output = llm.with_structured_output(Response)

    prompt = PromptTemplate(
        template="Generate a response based on the following context. " \
        "Here is the context: {context}. " \
        "Here is the question: {query}.",
        input_variables=["context", "query"]
    )

    query = state["messages"][0].content
    context = state["messages"][-1].content

    # FIX: Format the prompt as a string
    formatted_prompt = prompt.format(context=context, query=query)
    generated_response = llm_with_structured_output.invoke(formatted_prompt)

    return {"messages": [HumanMessage(content=generated_response.response)]}

def rewrite_response(state: State):
    """
    Transform the query to produce a better question.
    """
    class Rewrite(BaseModel):
        rewritten_query: str = Field(
            ...,
            description="The re-phrased question."
        )

    llm_with_structured_output = llm.with_structured_output(Rewrite)

    prompt = PromptTemplate(
        template="Rewrite the following question to produce a better question. " \
        "Here is the question: {query}.",
        input_variables=["query"]
    )

    query = state["messages"][0].content

    # FIX: Format the prompt as a string
    formatted_prompt = prompt.format(query=query)
    rewritten_query = llm_with_structured_output.invoke(formatted_prompt)

    return {"messages": [HumanMessage(content=rewritten_query.rewritten_query)]}

def create_workflow():
    # Define the state graph
    workflow = StateGraph(State)

    # Add nodes for each step
    workflow.add_node("agent", agent)
    workflow.add_node("grade_document", grade_document)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("rewrite_response", rewrite_response)

    # Define the flow between nodes
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "grade_document")
    workflow.add_conditional_edges(
        "grade_document",
        lambda result: result["next"],
        {
            "generate": "generate_response",
            "rewrite": "rewrite_response"
        }
    )
    workflow.add_edge("generate_response", END)
    workflow.add_edge("rewrite_response", "agent")

    # Compile the workflow
    graph = workflow.compile()

    return graph

# Create the workflow
workflow = create_workflow()

if __name__ == "__main__":

    # Example test query
    test_query = "What is langchain"

    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=test_query)]
    }

    # Run the workflow
    for output in workflow.stream(initial_state):
        print(output)