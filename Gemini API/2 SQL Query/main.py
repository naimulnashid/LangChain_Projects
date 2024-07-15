# Importing Necessary Libraries
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from few_shots import few_shots

from langchain_experimental.sql import SQLDatabaseChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.utilities import SQLDatabase
from langchain.vectorstores import Chroma


# Loading Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to Create Few-Shot Database Chain
def get_few_shot_db_chain():
    # Database connection details
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    # Connects to the SQL database using provided URI
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
    # Initializes the language model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", temperature=0.1)
    # Initializes the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # Prepares the data for vectorization
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    # Creates a vector store from texts and embeddings
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    # Selects examples based on semantic similarity
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    # A prompt template for generating MySQL queries
    mysql_prompt = """
    You are a MySQL expert. Given a question, first generate a syntactically correct MySQL query to execute. Then, review the query results and provide an answer to the question.
    If the user does not specify a certain number of examples, limit the query to return a maximum of {top_k} results using the LIMIT clause in MySQL. You can order the results to highlight the most relevant data in the database.
    Avoid querying all columns from a table. Only select the columns necessary to answer the question, and wrap each column name in backticks (`) to mark them as delimited identifiers.
    Ensure you only use column names visible in the tables provided. Do not query non-existent columns and be mindful of which columns belong to which tables.
    If the question includes "today," use the CURDATE() function to get the current date.
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"]
    )
    
    # Creates a chain for SQL database queries using the language model and prompt
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain

# Function to Handle User Input and Retrieve Answers
def user_input(user_question):
    chain = get_few_shot_db_chain()
    response = chain.run(user_question)
    return response


## Frontend

# Streamlit App Setup
st.set_page_config(page_title="Database Q&A", page_icon="🔍")
st.title("T Shirts: Database Q&A 👕")


# Initialize empty list to store question-answer pairs
if 'history' not in st.session_state:
    st.session_state.history = []

# Input Field for Questions
user_question = st.text_input("Ask a Question: ", key="question_input")

# Handling User Questions
if user_question:
    try:
        # Get the answer for the current question
        answer = user_input(user_question)

        # Add current question-answer pair to history
        st.session_state.history.append((user_question, answer))

        # Display all previous and current answers
        st.header("Questions & Answers")
        for idx, (question, answer) in enumerate(reversed(st.session_state.history)):  # Display in reverse order
            st.subheader(f"Q&A {len(st.session_state.history) - idx}")
            st.markdown(f"<span style='font-size:20px'>Question: {question}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:19px'>Answer: {answer}</span>", unsafe_allow_html=True)
            st.markdown("---")  # Separator between Q&A

    except Exception as e:
        st.error(f"An error occurred: {e}")