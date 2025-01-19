# LangChain Projects

1. **CSV Query**: Provides coherent answers to frequently asked questions using CSV files as a knowledge base.
2. **SQL Query**: Translates natural language questions into MySQL queries for database execution.
3. **PDF Query**: Conducts question-and-answer tasks based on information extracted from provided PDFs.
4. **ATS**: Matches resumes with job descriptions, highlighting missing key points to improve resumes.

---

### Key Features and Implementation Details

- **Context-Aware Reasoning Application**: 
  - Built using the LangChain framework.
  - Integrated with Gemini API for access to the Gemini-Pro Large Language Model.
  
- **Vectorization and Similarity Measurement**:
  - Utilized **GoogleGenerativeAIEmbeddings** for text vectorization.
  - Implemented **FAISS** and **Chroma** as vector databases for information storage.
  
- **Retriever Configuration**:
  - Enhanced LLM output by integrating external data.
  - Applied **Few-Shot Learning** for handling complex queries beyond standard Gemini capabilities.
  
- **Chains and Parsers**:
  - Constructed chains to manage prompts, user inquiries, conversation history, and context for LLM.
  - Used parsers to generate structured output responses.

- **Interactive Application Deployment**:
  - Developed an interactive interface using the **Streamlit** framework.
