# Importing Necessary Libraries
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Loading Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Initialize Google Generative AI & Embeddings
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", temperature=0.5)
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# File Paths
file_path = 'faqs.csv'
vectordb_file_path = "faiss_index"


# Function to create Vector Database
def create_vector_db():
    #  Loads the data from the specified CSV file
    loader = CSVLoader(file_path=file_path, source_column="prompt")
    data = loader.load()
    # Creates a vector database from the loaded data and embeddings, and then saves it locally
    vectordb = FAISS.from_documents(documents=data, embedding=embedding)
    vectordb.save_local(vectordb_file_path)


# Function to Load Conversational QA Chain
def get_conversational_chain():
    prompt_template = """
    Generate an answer based solely on the provided context and question. 
    Try to use as much text as possible from the "response" section of the source document without altering it significantly. 
    If the answer isn't found within the context, acknowledge with "I don't know" without attempting to fabricate an answer.

    CONTEXT: {context}

    QUESTION: {question}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to Handle User Input and Retrieve Answers
def user_input(user_question):
    new_db = FAISS.load_local(vectordb_file_path, embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

## Frontend

# Streamlit App Setup
st.set_page_config(page_title="FAQs Q&A", page_icon="🔍")
st.title("FAQs Q&A 🌱")


# Button to Create Knowledge Base
button_create_kb = st.button("Create Knowledgebase")
if button_create_kb:
    try:
        create_vector_db()
        st.success("Knowledgebase created successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Initialize empty list to store question-answer pairs
if 'history' not in st.session_state:
    st.session_state.history = []


# Input Field for Questions
user_question = st.text_input("Ask a Question:", key="question_input")


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
            st.write_stream(f"Question: {question}")
            st.write_stream(f"Answer: {answer}")
            #st.markdown(f"<span style='font-size:20px'>Question: {question}</span>", unsafe_allow_html=True)
            #st.markdown(f"<span style='font-size:19px'>Answer: {answer}</span>", unsafe_allow_html=True)
            st.write("---")  # Separator between Q&A

    except Exception as e:
        st.error(f"An error occurred: {e}")
