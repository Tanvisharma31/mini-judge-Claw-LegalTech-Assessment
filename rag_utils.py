"""
RAG Utilities for MiniJudge

This module implements Retrieval-Augmented Generation (RAG) functionality:
- Text chunking using LangChain
- Embedding creation and storage using OpenAI and FAISS
- Question answering using retrieved context and LLM (OpenAI or HuggingFace)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import tempfile
import json
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import openai
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGProcessor:
    """Class for implementing RAG functionality for legal document Q&A."""
    
    def __init__(self, 
                 use_openai: bool = True,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 openai_model: str = "gpt-3.5-turbo",
                 hf_model: str = "google/flan-t5-large",
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the RAG processor.
        
        Args:
            use_openai: Whether to use OpenAI for LLM and embeddings (default: True)
            chunk_size: Size of text chunks for processing (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            openai_model: OpenAI model to use for QA (default: gpt-3.5-turbo)
            hf_model: HuggingFace model to use if OpenAI is not available
            embedding_model: Model to use for embeddings (default: text-embedding-ada-002)
        """
        self.use_openai = use_openai
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_model = openai_model
        self.hf_model = hf_model
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.metadata = []
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Initialize embedding model
        if use_openai:
            # Check for OpenAI API key
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not found in environment. Some features may not work.")
            
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
            logger.info(f"Using OpenAI for embeddings: {embedding_model}")
            
            # Initialize OpenAI client
            try:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.llm = OpenAI(model_name=openai_model)
                logger.info(f"Using OpenAI for LLM: {openai_model}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI LLM: {e}")
                self.use_openai = False
                
        # Fallback to HuggingFace if OpenAI is not available or not selected
        if not self.use_openai:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                logger.info("Using HuggingFace for embeddings")
                
                # Initialize HuggingFace pipeline
                pipe = pipeline(
                    "text2text-generation",
                    model=self.hf_model,
                    max_new_tokens=512
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
                logger.info(f"Using HuggingFace for LLM: {hf_model}")
            except Exception as e:
                logger.error(f"Error initializing HuggingFace components: {e}")
                raise RuntimeError("Failed to initialize either OpenAI or HuggingFace components")
    
    def process_text(self, text: str) -> None:
        """
        Process text by splitting into chunks and creating embeddings.
        
        Args:
            text: Text extracted from the document
        """
        logger.info(f"Processing text of length: {len(text)}")
        
        # Split text into chunks
        self.chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(self.chunks)} chunks")
        
        # Generate metadata for each chunk
        self.metadata = [{"chunk_id": i, "chars": len(chunk)} for i, chunk in enumerate(self.chunks)]
        
        # Create embeddings
        self._create_embeddings()
    
    def _create_embeddings(self) -> None:
        """Create embeddings for all chunks and build FAISS index."""
        if not self.chunks:
            logger.warning("No chunks to embed")
            return
        
        try:
            logger.info("Creating embeddings for chunks...")
            chunk_embeddings = self.embeddings.embed_documents(self.chunks)
            
            # Convert to numpy array for FAISS
            embeddings_array = np.array(chunk_embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings_array)
            
            logger.info(f"Created FAISS index with {self.faiss_index.ntotal} vectors of dimension {dimension}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def save_index(self, directory: str) -> None:
        """
        Save the FAISS index, chunks, and metadata to disk.
        
        Args:
            directory: Directory to save the files
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if self.faiss_index and self.chunks:
            try:
                # Save FAISS index
                faiss.write_index(self.faiss_index, os.path.join(directory, "faiss_index.bin"))
                
                # Save chunks and metadata
                with open(os.path.join(directory, "chunks.json"), "w") as f:
                    json.dump(self.chunks, f)
                
                with open(os.path.join(directory, "metadata.json"), "w") as f:
                    json.dump(self.metadata, f)
                
                logger.info(f"Saved index and data to {directory}")
            except Exception as e:
                logger.error(f"Error saving index: {e}")
        else:
            logger.warning("No index or chunks to save")
    
    def load_index(self, directory: str) -> bool:
        """
        Load the FAISS index, chunks, and metadata from disk.
        
        Args:
            directory: Directory to load the files from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
            
            # Load chunks and metadata
            with open(os.path.join(directory, "chunks.json"), "r") as f:
                self.chunks = json.load(f)
            
            with open(os.path.join(directory, "metadata.json"), "r") as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded index and data from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        if not self.faiss_index or not self.chunks:
            raise ValueError("No index or chunks available. Process a document first.")
        
        # Create query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.faiss_index.search(query_array, min(top_k, len(self.chunks)))
        
        # Get the relevant chunks
        relevant_chunks = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
        return relevant_chunks
    
    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question based on the processed document.
        
        Args:
            query: User question
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary with answer and relevant chunks
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
        context = "\n\n".join(relevant_chunks)
        
        # Create the prompt
        prompt = f"""
        You are MiniJudge, a legal assistant that helps analyze court judgments.
        
        Below is an excerpt from a legal judgment document:
        
        {context}
        
        Based on the above excerpt, please answer the following question concisely and accurately.
        If the information is not available in the excerpt, state that clearly.
        
        Question: {query}
        
        Answer:
        """
        
        # Get answer from LLM
        try:
            if self.use_openai:
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are MiniJudge, a legal assistant that helps analyze court judgments."},
                        {"role": "user", "content": f"Based on the following excerpt from a legal judgment document, please answer this question: {query}\n\nExcerpt:\n{context}"}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                answer = response.choices[0].message.content
            else:
                answer = self.llm(prompt)
            
            return {
                "answer": answer,
                "source_chunks": relevant_chunks,
                "query": query
            }
        except Exception as e:
            logger.error(f"Error getting answer from LLM: {e}")
            return {
                "answer": f"Error getting answer: {str(e)}",
                "source_chunks": relevant_chunks,
                "query": query
            }


# Function to check available models/providers
def check_available_models() -> Dict[str, bool]:
    """
    Check which models/providers are available.
    
    Returns:
        Dictionary of available model providers
    """
    availability = {
        "openai": False,
        "huggingface": False
    }
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # Minimal test to verify API key works
            openai.Model.list()
            availability["openai"] = True
            logger.info("OpenAI API is available")
        except Exception as e:
            logger.warning(f"OpenAI API not available: {e}")
    
    # Check HuggingFace (simple check if importable)
    try:
        import transformers
        availability["huggingface"] = True
        logger.info("HuggingFace is available")
    except ImportError:
        logger.warning("HuggingFace not available")
    
    return availability


# Example usage
if __name__ == "__main__":
    # This is just for demonstration and testing
    test_text = """
    IN THE SUPREME COURT OF JUDICATURE
    COURT OF APPEAL (CIVIL DIVISION)
    ON APPEAL FROM THE HIGH COURT OF JUSTICE
    CHANCERY DIVISION
    Royal Courts of Justice
    London WC2A 2LL
    
    Date: 23rd May 2001
    
    Before:
    LORD JUSTICE ALDOUS
    LORD JUSTICE CHADWICK
    and
    LORD JUSTICE CLARKE
    
    This is a test document for the MiniJudge system.
    The purpose of this test is to demonstrate text splitting and retrieval.
    """
    
    availability = check_available_models()
    print(f"Available models: {availability}")
    
    # Initialize RAG processor based on availability
    processor = RAGProcessor(use_openai=availability["openai"])
    
    # Process test text
    processor.process_text(test_text)
    
    # Test retrieval
    result = processor.answer_question("Who were the judges in this case?")
    print(f"\nQuestion: Who were the judges in this case?")
    print(f"Answer: {result['answer']}")