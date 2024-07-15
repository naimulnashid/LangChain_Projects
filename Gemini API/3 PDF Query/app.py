# Importing Libraries
import os
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# Load Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


## Backend Functions

# Extracting Text from PDFs
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


# Splitting Text into Chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000, 
        chunk_overlap=50)
    chunks=text_splitter.split_text(text)
    return chunks


# Creating a Vector Store
def get_vector_store(text_chunks):
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index")


# Creating a Conversational Chain
def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context. 
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain


# Handling User Input
def user_input(user_question):
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()
    response=chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

## Frontend

# Streamlit App Setup
st.set_page_config(page_title="PDF Query", page_icon="📄")
st.title("PDF Query")


# Sidebar Menu
with st.sidebar:
    st.title("Menu")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

    if st.button("Submit"):
        if pdf_docs:
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload at least one PDF file.")


# Initialize empty list to store question-answer pairs
if 'history' not in st.session_state:
    st.session_state.history = []


# Input Field for Questions
user_question = st.text_input("Ask a Question in context to the provided PDFs: ", key="question_input")


# Handling User Questions
if user_question:
    try:
        # Get the answer for the current question using your backend function
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
