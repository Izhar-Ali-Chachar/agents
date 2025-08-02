from itertools import chain
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from typing import List, TypedDict
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Java-related URLs (you can add more)
urls = [
    # "https://www.geeksforgeeks.org/java/",
    "https://www.oracle.com/java/technologies/javase-downloads.html",
    "https://en.wikipedia.org/wiki/Java_(programming_language)"
]

# Load documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# Initialize embedding model and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(doc_splits, embedding_model)

# Set up retriever and LLM
retriever = vectorstore.as_retriever()
java_retriever_tool = create_retriever_tool(
    retriever,
    "retriever_vector_db_java",
    "Useful for retrieving Java-related documentation based on a query."
)

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class State(TypedDict):
    question: str
    generation: str
    documents: List[str]
    web_search: str

def retrieve(state: State):
    """
    Retrieves relevant documents based on the user's query.

    args:
        state (State): The current state containing the user's question.
    returns:
        dict: A dictionary containing the retrieved documents and the original question.
    """

    query = state['question']

    documents = retriever.invoke(query)

    return {'documents': documents, 'question': query}

def generate(state: State):
    """
    Generates a response based on the retrieved documents and the user's question.

    args:
        state (State): The current state containing the user's question and retrieved documents.
    returns:
        dict: A dictionary containing the generated response.
    """

    documents = state['documents']
    question = state['question']

    # Combine all document contents into a single context string
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = hub.pull("rlm/rag-prompt")

    parser = StrOutputParser()

    chain = prompt | llm | parser

    generate = chain.invoke({
        'question': question,
        'context': context
    })

    return {'generation': generate, 'question': question, 'documents': documents}

def document_grader(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents 
    """

    class DocumentGrader(BaseModel):
        """
        Model to grade the relevance of documents.
        """
        binary_score: str = Field(
            ...,
            description="The relevance score of the documents."
        )

    llm_with_structured_output = llm.with_structured_output(DocumentGrader)

    system_message="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Here is the question: {question}. \n Here is the document: {document}. \n Please provide a binary score 'yes' or 'no' indicating relevance.")
    ])

    grade_retriever = prompt | llm_with_structured_output

    question = state['question'] 
    
    relevant_docs = []
    web_search = "No"
    for doc in state['documents']:

        grade = grade_retriever.invoke({
            'question': question,
            'document': doc
        })

        if grade.binary_score == 'yes':
            relevant_docs.append(doc)
        else:
            web_search = "Yes"

    return {"documents": relevant_docs, "web_search": web_search, "question": question}

def transfrom_question(state):
    """
    Transforms the user's question to be more suitable for retrieval.

    input:
        state (State): The current state containing the user's question.

    output:
        dict: A dictionary containing the transformed question.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a question transformer. Transform the user's question to be more suitable for retrieval."),
        ("user", "Here is the question: {question}. Please transform it.")
    ])

    question = state['question']
    document = state['documents']

    system = """You are a question transformer. Transform the user's question to be more suitable for retrieval.
    Here is the question: {question}. Please transform it."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Here is the question: {question}. Please transform it.")
    ])

    question_rewriter = prompt | llm | StrOutputParser()

    transformed_question = question_rewriter.invoke({
        'question': question
    })

    return {"doc": document, "question": transformed_question}

search_tool = TavilySearchResults(
    query="",
    num_results=3,
    api_key=os.getenv("TAVILY_API_KEY")
)

def web_search(state):
    """
     Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    query = state['question']

    docs = search_tool.invoke(query)
    web_results = "\n".join([doc.page_content for doc in docs])
    web_results = Document(page_content=web_results, metadata={"source": "web_search"})
    state['documents'].append(web_results)

    return {"documents": state['documents'], "question": query}


def decide_to_generate(state: State):
    """
    Decide whether to generate a response or rewrite the question based on the retrieved documents.

    args:
        state (State): The current state containing the user's question and retrieved documents.
    returns:
        dict: A dictionary indicating the next step in the workflow.
    """
    
    if state['web_search'] == "yes":
        return "rewrite"
    else:
        return "generate"


from langgraph.graph import StateGraph, END

def corrective_rag_graph():
    """
    Creates a state graph for the corrective RAG workflow.

    Returns:
        StateGraph: The state graph for the corrective RAG workflow.
    """
    
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("document_grader", document_grader)
    graph.add_node("transfrom_question", transfrom_question)
    graph.add_node("web_search", web_search)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "document_grader")
    graph.add_conditional_edges(
        "document_grader",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "transfrom_question"
        }
    )
    graph.add_edge("transfrom_question", "web_search")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

if __name__ == "__main__":
    # Create the corrective RAG graph
    graph = corrective_rag_graph()
    print("Corrective RAG graph created successfully.")
    # You can now use this graph in your application
    # For example, you can run it with an initial state
    initial_state = {
        "question": "What is Java?",
        "documents": [],
        "web_search": "No"
    }
    for output in graph.stream(initial_state):
        print(output)
    print("Workflow completed.")