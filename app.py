'''import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Configuration - Replace these with your actual API keys

GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDtAHVhkdISWedvCTlgC0lm0HA0PqpvKLU"
# Get from https://makersuite.google.com

# Set up the Streamlit app with professional styling
st.set_page_config(
    page_title="Professional PDF QA System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stTextInput>div>div>input {
        padding: 10px;
        border-radius: 5px;
    }
    .stFileUploader>div>div>div>button {
        padding: 10px 24px;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📄 Professional Document Analysis System")
st.markdown("""
<div style="background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:20px">
    <h3 style="color:#2c3e50;margin-top:0;">Upload documents and get AI-powered answers</h3>
    <p style="color:#7f8c8d;">Powered by Groq's fastest LLMs and Google's AI embeddings</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.file_name = None
    st.session_state.chat_history = []


# Function to process the uploaded PDF
def process_pdf(uploaded_file):
    try:
        with st.status("Processing document...", expanded=True) as status:
            st.write("Saving temporary file...")
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            st.write("Extracting text from PDF...")
            # Extract text from PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()

            # Combine all pages into a single text
            text = "\n".join([page.page_content for page in pages])

            # Clean up the temporary file
            os.unlink(tmp_file_path)

            st.write("Splitting text into chunks...")
            # Split text into chunks optimized for RAG
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Slightly larger chunks for better context
                chunk_overlap=300,  # More overlap for better continuity
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)

            st.write("Creating semantic index...")
            # Create embeddings using Google's latest model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            # Create FAISS vector store with efficient indexing
            vector_store = FAISS.from_texts(
                chunks,
                embedding=embeddings,
                metadatas=[{"source": f"chunk-{i}"} for i in range(len(chunks))]
            )

            st.write("Initializing AI model...")
            # Set up the RAG chain with Groq's fastest available model
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama3-70b-8192",  # Most powerful available model
                temperature=0.3,  # Balanced between creativity and factuality
                max_tokens=2048  # Allow for detailed responses
            )

            # Professional-grade prompt template
            prompt_template = """
            You are a professional document analysis assistant. Your task is to provide accurate, 
            well-structured answers based strictly on the provided document context.

            Document Context:
            {context}

            Question: {question}

            Guidelines:
            1. Answer concisely but thoroughly
            2. Cite relevant document sections when possible
            3. If unsure, say "The document doesn't contain clear information about this"
            4. Format complex answers with bullet points or numbered lists
            5. Maintain professional tone

            Answer:
            """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="mmr",  # Max marginal relevance for better diversity
                    search_kwargs={"k": 4}  # Retrieve 4 most relevant chunks
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            # Update session state
            st.session_state.pdf_processed = True
            st.session_state.vector_store = vector_store
            st.session_state.qa_chain = qa_chain
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

            status.update(label="Document processing complete!", state="complete", expanded=False)
            st.toast("Document ready for analysis", icon="✅")

    except Exception as e:
        st.error(f"Professional error handling: {str(e)}")
        st.session_state.pdf_processed = False


# Function to handle question answering with professional formatting
def answer_question(question):
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a document first.")
        return

    try:
        with st.spinner("Analyzing document..."):
            # Add user question to chat history
            st.session_state.chat_history.append(("user", question))

            # Get answer from RAG chain
            result = st.session_state.qa_chain({"query": question})
            answer = result["result"]

            # Add AI response to chat history
            st.session_state.chat_history.append(("ai", answer))

            # Display chat in a professional manner
            st.subheader("Analysis Results")
            with st.container():
                for sender, message in st.session_state.chat_history:
                    if sender == "user":
                        st.markdown(f"""
                        <div style='background-color:#e3f2fd;padding:10px;border-radius:5px;margin-bottom:10px;'>
                            <strong>You:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color:#f5f5f5;padding:10px;border-radius:5px;margin-bottom:20px;'>
                            <strong>AI Analyst:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)

            # Show source documents in an expandable section
            if "source_documents" in result and result["source_documents"]:
                with st.expander("🔍 View Document References", expanded=False):
                    st.subheader("Relevant Document Sections")
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"""
                        <div style='background-color:#fff8e1;padding:10px;border-radius:5px;margin-bottom:10px;'>
                            <strong>Reference {i}:</strong><br>
                            {doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")


# Main app interface - Professional layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None and not st.session_state.pdf_processed:
        process_pdf(uploaded_file)

    if st.session_state.pdf_processed:
        st.success(f"Analyzing: {st.session_state.file_name}")
        question = st.text_input(
            "Enter your question about the document:",
            placeholder="E.g., What are the key findings in this report?",
            key="question_input"
        )

        if st.button("Submit Question", type="primary") and question:
            answer_question(question)

with col2:
    st.subheader("System Information")
    st.markdown("""
    **Current Model:**  
    LLaMA 3 70B (8192 context)  

    **Embeddings:**  
    Google Gemini 1.0  

    **Vector Store:**  
    FAISS (Facebook AI)  

    **Processing:**  
    - Chunk size: 1500 chars  
    - Overlap: 300 chars  
    - Retrieval: 4 best matches  
    """)

    if st.session_state.pdf_processed:
        if st.button("New Document", help="Clear current analysis and start over"):
            st.session_state.pdf_processed = False
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.file_name = None
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#7f8c8d;font-size:0.9em;">
    <p>Professional Document Analysis System v1.0</p>
    <p>For optimal results, upload clear PDF documents with readable text</p>
</div>
""", unsafe_allow_html=True)'''

import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Configuration - Replace these with your actual API keys
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDtAHVhkdISWedvCTlgC0lm0HA0PqpvKLU"

# Set up the Streamlit app with professional styling
st.set_page_config(
    page_title="Professional Document Analysis",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4A6FA5;
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 500;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3A5A8C;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input {
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
    .stFileUploader>div>div>div>button {
        padding: 12px 24px;
        border-radius: 8px;
        width: 100%;
    }
    .css-1aumxhk {
        background-color: #F8F9FA;
        border-radius: 12px;
        padding: 20px;
    }
    .chat-message {
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        line-height: 1.6;
    }
    .user-message {
        background-color: #EFF6FF;
        border-left: 4px solid #4A6FA5;
    }
    .ai-message {
        background-color: #F8F9FA;
        border-left: 4px solid #6C757D;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        color: #2C3E50;
        font-weight: 600;
    }
    .header p {
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("""
<div class="header">
    <h1 style="
        font-size: 2.8rem;
        font-weight: 700;
        color: #1F3A93;
        margin-bottom: 0.2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        📄 Document Intelligence Suite
    </h1>
    <p style="
        font-size: 1.1rem;
        color: #5D6D7E;
        font-weight: 400;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        AI-Powered Precision Analysis for Your Uploaded Documents
    </p>
</div>
""", unsafe_allow_html=True)


# Initialize session state variables
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.file_name = None
    st.session_state.chat_history = []


# Function to process the uploaded PDF
def process_pdf(uploaded_file):
    try:
        with st.status("Processing document...", expanded=True) as status:
            st.write("Saving temporary file...")
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            st.write("Extracting text from PDF...")
            # Extract text from PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()

            # Combine all pages into a single text
            text = "\n".join([page.page_content for page in pages])

            # Clean up the temporary file
            os.unlink(tmp_file_path)

            st.write("Splitting text into chunks...")
            # Split text into chunks optimized for RAG
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)

            st.write("Creating semantic index...")
            # Create embeddings using Google's latest model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            # Create FAISS vector store with efficient indexing
            vector_store = FAISS.from_texts(
                chunks,
                embedding=embeddings,
                metadatas=[{"source": f"chunk-{i}"} for i in range(len(chunks))]
            )

            st.write("Initializing AI model...")
            # Set up the RAG chain with Groq's fastest available model
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama3-70b-8192",
                temperature=0.3,
                max_tokens=2048
            )

            # Professional-grade prompt template
            prompt_template = """
            You are a professional document analysis assistant. Your task is to provide accurate, 
            well-structured answers based strictly on the provided document context.

            Document Context:
            {context}

            Question: {question}

            Guidelines:
            1. Answer concisely but thoroughly
            2. Cite relevant document sections when possible
            3. If unsure, say "The document doesn't contain clear information about this"
            4. Format complex answers with bullet points or numbered lists
            5. Maintain professional tone

            Answer:
            """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

            # Update session state
            st.session_state.pdf_processed = True
            st.session_state.vector_store = vector_store
            st.session_state.qa_chain = qa_chain
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

            status.update(label="Document processing complete!", state="complete", expanded=False)
            st.toast("Document ready for analysis", icon="✅")

    except Exception as e:
        st.error(f"An error occurred while processing the document: {str(e)}")
        st.session_state.pdf_processed = False


# Function to handle question answering with professional formatting
def answer_question(question):
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a document first.")
        return

    try:
        with st.spinner("Analyzing document..."):
            # Add user question to chat history
            st.session_state.chat_history.append(("user", question))

            # Get answer from RAG chain
            result = st.session_state.qa_chain({"query": question})
            answer = result["result"]

            # Add AI response to chat history
            st.session_state.chat_history.append(("ai", answer))

            # Display chat in a professional manner
            st.subheader("Document Analysis")
            with st.container():
                for sender, message in st.session_state.chat_history:
                    if sender == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>Your question:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message ai-message">
                            <strong>Analysis:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)

            # Show source documents in an expandable section
            if "source_documents" in result and result["source_documents"]:
                with st.expander("🔍 View supporting document excerpts", expanded=False):
                    st.subheader("Relevant Document Sections")
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"""
                        <div style='background-color:#fff8e1;padding:12px;border-radius:8px;margin-bottom:12px;'>
                            <strong>Excerpt {i}:</strong><br>
                            {doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")


# Main app interface
st.subheader("Upload Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None and not st.session_state.pdf_processed:
    process_pdf(uploaded_file)

if st.session_state.pdf_processed:
    st.success(f"Active document: {st.session_state.file_name}")
    question = st.text_input(
        "Enter your question about the document:",
        placeholder="What information would you like to extract from this document?",
        key="question_input"
    )

    if st.button("Analyze Document", type="primary") and question:
        answer_question(question)

    if st.button("Clear Document", type="secondary"):
        st.session_state.pdf_processed = False
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.file_name = None
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#7f8c8d;font-size:0.9em;margin-top:2rem;">
    <p>Professional Document Analysis System • Secure Processing • Your data is never stored</p>
</div>
""", unsafe_allow_html=True)

