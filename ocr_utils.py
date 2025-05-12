"""
OCR Utilities for MiniJudge

This module handles the extraction of text from PDF and image files using Tesseract OCR.
It can process both text-based PDFs (via direct extraction) and image-based PDFs or images 
(via OCR).
"""

import os
import io
import tempfile
from typing import Optional, List, Union
import logging

import pytesseract
from PIL import Image
import pdf2image
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRProcessor:
    """Class for extracting text from PDFs and images using OCR when necessary."""
    
    def __init__(self, tesseract_path: Optional[str] = None, language: str = 'eng'):
        """
        Initialize the OCR processor.
        
        Args:
            tesseract_path: Path to the Tesseract executable (optional)
            language: Language for OCR (default: 'eng' for English)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.language = language
        logger.info(f"OCRProcessor initialized with language: {language}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.process_pdf(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            return self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def extract_text_from_bytes(self, file_bytes: bytes, file_name: str) -> str:
        """
        Extract text from file bytes.
        
        Args:
            file_bytes: Bytes of the file
            file_name: Name of the file (for determining type)
            
        Returns:
            Extracted text as a string
        """
        file_extension = os.path.splitext(file_name)[1].lower()
        
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        
        try:
            text = self.extract_text_from_file(temp_path)
            return text
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF file, combining direct text extraction and OCR as needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # First, try direct text extraction
        direct_text = self._extract_text_directly(pdf_path)
        
        # Check if we got meaningful text
        if direct_text and len(direct_text.strip()) > 100:  # Arbitrary threshold
            logger.info("Successfully extracted text directly from PDF")
            return direct_text
        
        # If direct extraction didn't yield good results, use OCR
        logger.info("Direct extraction yielded insufficient text, falling back to OCR")
        return self._extract_text_with_ocr(pdf_path)
    
    def _extract_text_directly(self, pdf_path: str) -> str:
        """
        Extract text directly from a PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            text_content = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text_content.append(page.get_text())
            return "\n".join(text_content)
        except Exception as e:
            logger.warning(f"Error in direct text extraction: {e}")
            return ""
    
    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from a PDF using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            # Process each page
            text_content = []
            for i, img in enumerate(images):
                logger.info(f"OCR processing page {i+1}/{len(images)}")
                text = pytesseract.image_to_string(img, lang=self.language)
                text_content.append(text)
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {e}")
            raise
    
    def process_image(self, image_path: str) -> str:
        """
        Process an image file using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
        """
        logger.info(f"Processing image: {image_path}")
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=self.language)
            return text
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

# Function to check if the required dependencies are installed
def check_dependencies() -> bool:
    """
    Check if all required dependencies are properly installed.
    
    Returns:
        Boolean indicating if all dependencies are available
    """
    try:
        # Check Tesseract
        pytesseract.get_tesseract_version()
        
        # Check PyMuPDF
        fitz.version
        
        # Check pdf2image
        # This is a simple check - pdf2image depends on poppler
        assert pdf2image.__version__
        
        return True
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # This is just for demonstration and testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_utils.py <path_to_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not check_dependencies():
        print("Some dependencies are missing. Please check the installation.")
        sys.exit(1)
    
    processor = OCRProcessor()
    try:
        text = processor.extract_text_from_file(file_path)
        print(f"Successfully extracted {len(text)} characters of text")
        print("\nFirst 500 characters:")
        print(text[:500] + "...")
    except Exception as e:
        print(f"Error processing file: {e}")