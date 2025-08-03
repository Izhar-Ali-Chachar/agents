from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()

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


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, max_output_tokens=1024)

print(llm.invoke("Hello, world!"))

class State(TypedDict):
    question: str
    documents: List[str]
    generation: str

def route_question(state: State) -> str:
    """Route the question to either web search or retrieval based on its content."""

    class RouteQuery(BaseModel):
        """Route the question to either web search or retrieval based on its content."""

        datasource: Literal["web_search", "retreive"] = Field(
            description="The datasource to use for the question."
        )
    
    llm_with_structured_output = llm.with_structured_output(RouteQuery)

    question = state["question"]

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    formatted_prompt = prompt.format(question=question)
    response = llm_with_structured_output.invoke(formatted_prompt)

    if response.datasource == "web_search":
        return "web_search"
    elif response.datasource == "retreive":
        return "retreive"
    
def retreive(state: State) -> State:
    """Retrieve relevant documents based on the user's question.
    Args:
        state (State): The current state containing the user's question.
    Returns:
        State (dict): The updated state with retrieved documents.
    """

    query = state['question']

    documents = retriever.invoke(query)

    return {'documents': documents, 'question': query}
    
def generate(state: State) -> State:
    """Generate a response based on the retrieved documents and the user's question.
    Args:
        state (State): The current state containing the user's question and retrieved documents.
    Returns:
        State (dict): The updated state with the generated response.
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

def transform_query(state: State) -> State:
    """Transform the user's question into a more effective query for retrieval."""
    question = state["question"]
    documents = state["documents"]

    # Combine all document contents into a single context string
    context = "\n\n".join([doc.page_content for doc in documents])

    system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    transformed_query = chain.invoke({
        'question': question,
        'context': context
    })

    return {'question': transformed_query, 'documents': documents}

def grade_documents(state: State) -> State:
    """Determine whether the retrieved documents are relevant to the question.
    Args:
        state (State): The current state containing the user's question and retrieved documents.
    Returns:
        State (dict): The updated state with only relevant documents.
    """

    class DocumentGrader(BaseModel):
        """Model to grade the relevance of documents."""
        binary_score: str = Field(
            ...,
            description="The relevance score of the documents."
        )

    llm_with_structured_output = llm.with_structured_output(DocumentGrader)

    system_message = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}"),
        ("assistant", "{documents}")
    ])

    retriever_grader = prompt | llm_with_structured_output

    question = state["question"]
    documents = state["documents"]

    filter_documents = []
    for doc in documents:
        grade = retriever_grader.invoke({
            "question": question,
            "documents": [doc]
        })

        if grade.binary_score == 'yes':
            filter_documents.append(doc)
        else:
            continue

    return {"question": question, "documents": filter_documents}

def web_search(state: State) -> State:
    """Perform a web search to find relevant documents for the user's question.
    Args:
        state (State): The current state containing the user's question.
    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]

    search_tool = TavilySearchResults()
    docs = search_tool.invoke({
        "query": question,
        "num_results": 5,
        "search_engine": "google"
    })
    web_results = '\n\n'.join([d['content'] for d in docs])
    web_results = Document(page_content=web_results, metadata={"source": "web_search"})

    return {"documents": [web_results], "question": question}

def decide_to_generate(state: State) -> str:
    """Decide whether to generate a response based on the relevance of the documents."""
    
    question = state['question']
    filtered_docs = state['documents']

    if not filtered_docs:
        return "transform_query"
    else:
        return "generate"
    
def grade_generation_v_documents_and_question(state: State) -> str:
    """Grade the generated response against the question and retrieved documents."""
    
    class GenerationGrader(BaseModel):
        """Model to grade the generation against the question and documents."""
        score: Literal["useful", "not useful", "not supported"] = Field(
            ...,
            description="The usefulness of the generated response."
        )

    llm_with_structured_output = llm.with_structured_output(GenerationGrader)

    system_message = """You are a grader assessing the usefulness of a generated response to a user question. \n 
    If the response is relevant and useful, grade it as 'useful'. \n
    If it is not useful, grade it as 'not useful'. \n
    If the generation is not supported by the documents, grade it as 'not supported'."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Here is the question: {question}. \n Here are the documents: {documents}. \n Here is the generated response: {generation}. \n Please provide a score 'useful', 'not useful', or 'not supported'.")
    ])

    grader = prompt | llm_with_structured_output

    question = state['question']
    documents = state['documents']
    generation = state['generation']

    grade = grader.invoke({
        'question': question,
        'documents': documents,
        'generation': generation
    })

    return grade.score

def create_state_graph():
    graph = StateGraph(State)
    # Define states and transitions
    graph.add_node("web_search", web_search)
    graph.add_node("retreive", retreive)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("transform_query", transform_query)

    graph.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "retreive": "retreive",
        }
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("retreive", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        }
    )

    graph.add_edge("transform_query", "retreive")
    graph.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return graph.compile()

if __name__ == "__main__":
    graph = create_state_graph()

    initial_state = {
        "question": "What is the latest version of Java?",
        "documents": [],
        "generation": ""
    }

    for output in graph.stream(initial_state):
        print(output)
        if output.get("generation"):
            print(f"Generated response: {output['generation']}")
        if output.get("documents"):
            print(f"Retrieved documents: {output['documents']}")
        if output.get("web_search"):
            print(f"Web search results: {output['web_search']}")
        if output.get("question"):
            print(f"Transformed question: {output['question']}")
        if output.get("score"):
            print(f"Grading score: {output['score']}")
        if output.get("documents"):
            print(f"Filtered documents: {output['documents']}")
        if output.get("state"):
            print(f"Current state: {output['state']}")
        if output.get("next_step"):
            print(f"Next step: {output['next_step']}")
        if output.get("end"):
            print("Workflow completed.")
            break
    print("Adaptive RAG graph created successfully.")
    # You can now use this graph in your application