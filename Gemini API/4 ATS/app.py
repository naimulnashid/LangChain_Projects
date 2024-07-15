# Importing Necessary Libraries
import os
import json
import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
from dotenv import load_dotenv


# Loading Environment Variables
load_dotenv()

# Configuring Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


## Defining Helper Functions

# Function to Get Response from Generative AI Model
def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-1.5-pro-001')
    response = model.generate_content(input)
    return response.text


# Function to Extract Text from Uploaded PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text


# Prompt Template
input_prompt="""
Function as a proficient and seasoned Application Tracking System. Evaluate the resume against the provided job description, taking into account the highly competitive job market. Aim to offer optimal guidance for enhancing the resume, determining the match percentage based on the job description and identifying missing keywords.

resume: {resume_text}
description: {job_description}

I want the response in one single string having the structure
{{"JD Match": "%", "Missing Keywords": [], "Profile Summary": ""}}
"""


## Streamlit App

# Setting Up Streamlit App Interface
st.set_page_config(page_title="Smart ATS", page_icon="📄")
st.title("Smart ATS")
st.write("Improve your resume by evaluating it against a job description.")


# Input Form
with st.form("resume_form"):
    st.subheader("Job Description")
    job_description = st.text_area("Paste the Job Description", help="Enter the job description you want to match your resume against.")
    
    st.subheader("Resume Upload")
    uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload your resume in PDF format.")

    submit = st.form_submit_button("Submit")


# Handling Form Submission
if submit:
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            try:
                resume_text = input_pdf_text(uploaded_file)
                formatted_prompt = input_prompt.format(resume_text=resume_text, job_description=job_description)
                response = get_gemini_response(formatted_prompt)
                
                response_json = json.loads(response)
                
                st.subheader("Evaluation Result")
                #st.success(f"**JD Match:** {response_json['JD Match']}")
                #st.success(f"**Missing Keywords:** {', '.join(response_json['Missing Keywords'])}")
                #st.success(f"**Profile Summary:** {response_json['Profile Summary']}")
                st.markdown(f"<span style='font-size:19px'><strong>JD Match:</strong> {response_json['JD Match']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:19px'><strong>Missing Keywords:</strong> {', '.join(response_json['Missing Keywords'])}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:19px'><strong>Profile Summary:</strong> {response_json['Profile Summary']}</span>", unsafe_allow_html=True)


            
            except json.JSONDecodeError:
                st.error("Error parsing the response. Please ensure the response is correctly formatted JSON.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload your resume.")
