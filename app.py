"""
MiniJudge - Streamlit App

A legal tech prototype that allows users to upload court judgments (PDF/images),
extract text using OCR, and answer natural language questions using RAG + LLM.
"""

import os
import io
import tempfile
import uuid
import time
from typing import Optional, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from ocr_utils import OCRProcessor, check_dependencies
from rag_utils import RAGProcessor, check_available_models

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MiniJudge - Legal Document QA",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None
if 'rag_processor' not in st.session_state:
    st.session_state.rag_processor = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Create a data directory for this session
SESSION_DIR = os.path.join("data", st.session_state.session_id)
os.makedirs(SESSION_DIR, exist_ok=True)

# Check dependencies and available models
dependencies_ok = check_dependencies()
available_models = check_available_models()

def process_document(uploaded_file) -> bool:
    """
    Process an uploaded document file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Boolean indicating success
    """
    try:
        # Update status
        st.session_state.processing_status = "Processing document..."
        st.session_state.document_name = uploaded_file.name
        
        # Initialize OCR processor
        ocr = OCRProcessor()
        
        # Get file bytes
        file_bytes = uploaded_file.getvalue()
        
        # Extract text using OCR
        start_time = time.time()
        extracted_text = ocr.extract_text_from_bytes(file_bytes, uploaded_file.name)
        ocr_time = time.time() - start_time
        
        if not extracted_text or len(extracted_text) < 50:
            st.error("Text extraction failed or yielded insufficient text. Please try another document.")
            st.session_state.processing_status = "Failed"
            return False
        
        # Store the extracted text
        st.session_state.ocr_text = extracted_text
        
        # Save extracted text to file
        with open(os.path.join(SESSION_DIR, "extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        # Initialize RAG processor
        use_openai = available_models.get("openai", False)
        rag = RAGProcessor(use_openai=use_openai)
        
        # Process text with the RAG system
        start_time = time.time()
        rag.process_text(extracted_text)
        rag_time = time.time() - start_time
        
        # Save the RAG index
        rag.save_index(SESSION_DIR)
        
        # Store the processor in session state
        st.session_state.rag_processor = rag
        st.session_state.document_processed = True
        st.session_state.processing_status = "Ready"
        
        # Log processing statistics
        st.session_state.stats = {
            "ocr_time": ocr_time,
            "rag_time": rag_time,
            "text_length": len(extracted_text),
            "num_chunks": len(rag.chunks) if hasattr(rag, "chunks") else 0
        }
        
        return True
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.session_state.processing_status = "Failed"
        return False

def answer_question(question: str) -> Dict[str, Any]:
    """
    Answer a question using the RAG system.
    
    Args:
        question: User question
        
    Returns:
        Dictionary with answer and source information
    """
    if not st.session_state.document_processed or st.session_state.rag_processor is None:
        return {
            "answer": "Please upload and process a document first.",
            "source_chunks": [],
            "query": question
        }
    
    try:
        return st.session_state.rag_processor.answer_question(question)
    except Exception as e:
        return {
            "answer": f"Error answering question: {str(e)}",
            "source_chunks": [],
            "query": question
        }

# UI Components

# Title and description
st.title("⚖️ MiniJudge")
st.markdown("""
A legal tech prototype that analyzes court judgments and answers your questions.
Upload a PDF or image of a judgment document to get started.
""")

# Dependency check warning
if not dependencies_ok:
    st.warning("Some dependencies are missing or not properly configured. OCR functionality may be limited.")

# Model availability info
with st.sidebar:
    st.header("System Information")
    st.write("**Available LLM Providers:**")
    for provider, available in available_models.items():
        status = "✅ Available" if available else "❌ Not available"
        st.write(f"- {provider.capitalize()}: {status}")
    
    if not any(available_models.values()):
        st.error("No LLM providers are available. Please check your API keys and dependencies.")
    
    st.write("**Processing Settings:**")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    st.write("**Session Info:**")
    if st.session_state.document_name:
        st.write(f"Document: {st.session_state.document_name}")
    if st.session_state.processing_status:
        st.write(f"Status: {st.session_state.processing_status}")
    
    st.write("---")
    if st.button("Reset Session"):
        # Clear session state and refresh
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a court judgment (PDF or image)", 
                                     type=["pdf", "png", "jpg", "jpeg"])
    
    process_button = st.button("Process Document")
    
    if uploaded_file and process_button:
        with st.spinner("Processing document... This may take a few minutes."):
            success = process_document(uploaded_file)
            if success:
                st.success(f"Document processed successfully: {uploaded_file.name}")
                if st.session_state.stats:
                    st.write(f"Text length: {st.session_state.stats['text_length']} characters")
                    st.write(f"Processing time: {st.session_state.stats['ocr_time']:.2f}s (OCR) + "
                             f"{st.session_state.stats['rag_time']:.2f}s (RAG)")
            else:
                st.error("Failed to process document")
    
    # Display the extracted text (collapsible)
    if st.session_state.ocr_text:
        with st.expander("View Extracted Text"):
            st.text_area("", st.session_state.ocr_text, height=400)

with col2:
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the judgment")
    ask_button = st.button("Ask")
    
    if ask_button and question:
        with st.spinner("Analyzing..."):
            result = answer_question(question)
            
            st.markdown("### Answer")
            st.write(result["answer"])
            
            with st.expander("View Source Chunks"):
                for i, chunk in enumerate(result["source_chunks"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(chunk)
    
    # Disclaimer 
    st.markdown("---")
    st.caption("""
    **Disclaimer:** MiniJudge is a prototype tool designed for research and educational purposes. 
    It is not a substitute for legal advice, and its outputs should not be considered legal opinions. 
    Always consult with a qualified legal professional for legal matters.
    """)

# Function to load a previous session
def load_session():
    """Load a previously processed document if available."""
    if os.path.exists(SESSION_DIR):
        # Check if we have extracted text
        text_path = os.path.join(SESSION_DIR, "extracted_text.txt")
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                st.session_state.ocr_text = f.read()
            
            # Try to load the RAG processor
            try:
                use_openai = available_models.get("openai", False)
                rag = RAGProcessor(use_openai=use_openai)
                if rag.load_index(SESSION_DIR):
                    st.session_state.rag_processor = rag
                    st.session_state.document_processed = True
                    st.session_state.processing_status = "Ready"
                    return True
            except Exception as e:
                st.error(f"Error loading previous session: {str(e)}")
    
    return False

# Try to load previous session on startup
if 'document_processed' not in st.session_state or not st.session_state.document_processed:
    load_session()

# Main function
def main():
    pass  # The main logic is already implemented above

if __name__ == "__main__":
    main()