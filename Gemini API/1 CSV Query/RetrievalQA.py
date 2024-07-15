# Importing Necessary Libraries
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


# Loading Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialize Google Generative AI
model=ChatGoogleGenerativeAI(model="gemini-1.5-pro-001", temperature=0.5)
# Initialize Google Generative AI Embeddings
embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# File Paths
file_path='faqs.csv'
vectordb_file_path = "faiss_index"


# Function to create Vector Database
def create_vector_db():
    #  Loads the data from the specified CSV file.
    loader = CSVLoader(file_path=file_path, source_column="prompt")
    data = loader.load()
    # Creates a vector database from the loaded data and embeddings, and then saves it locally.
    vectordb = FAISS.from_documents(documents=data, embedding=embedding)
    vectordb.save_local(vectordb_file_path)

#Function to Get QA Chain
def get_qa_chain():
    # Loads the vector database from the local file
    vectordb = FAISS.load_local(vectordb_file_path, embedding, allow_dangerous_deserialization=True)

    # Creates a retriever to query the vector database with a score threshold for relevance
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Initializes a RetrievalQA
    chain = RetrievalQA(retriever=retriever)
    
    return chain



# Streamlit App
st.set_page_config(page_title="FAQs Q&A", page_icon="🔍")
st.title("FAQs Q&A 🌱")

# Button to Create Knowledge Base
button = st.button("Create Knowledgebase")
if button:
    try:
        create_vector_db()
        st.success("Knowledgebase created successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Input Field for Questions
question = st.text_input("Question: ")

# Handling User Questions
if question:
    try:
        # Get the QA chain
        chain = get_qa_chain()

        # Get the response to the user's question
        response = chain(question)

        # Display the answer
        st.header("Answer")
        st.write(response["result"])
    except Exception as e:
        st.error(f"An error occurred: {e}")