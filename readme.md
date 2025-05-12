# MiniJudge

## Legal Document Q&A Using OCR and RAG

MiniJudge is a legal tech prototype that allows users to upload court judgments (PDF or images), extract text using OCR, and answer natural language questions about the content using Retrieval-Augmented Generation (RAG) and large language models.

## Features

- **Document Processing**: Extract text from PDF and image files using Optical Character Recognition (OCR)
- **Intelligent Text Extraction**: Automatically chooses between direct text extraction and OCR based on document type
- **Semantic Search**: Retrieves relevant document sections using vector similarity
- **AI-Powered Answers**: Generates accurate answers using large language models with context from the document
- **User-Friendly Interface**: Clean Streamlit UI for easy document upload and question asking
- **Provider Flexibility**: Works with either OpenAI or HuggingFace models

## System Architecture

MiniJudge consists of three main components:

1. **OCR Module (`ocr_utils.py`)**: 
   - Extracts text from PDFs and images using Tesseract OCR
   - Handles both image-based and text-based PDFs
   - Provides fallback options for different document types

2. **RAG Module (`rag_utils.py`)**: 
   - Splits text into semantic chunks
   - Creates and stores vector embeddings using OpenAI or HuggingFace
   - Retrieves relevant context for questions
   - Generates answers using LLMs with retrieved context

3. **Web Interface (`app.py`)**: 
   - Streamlit-based user interface
   - Document upload and processing
   - Question input and answer display
   - Session management and document persistence

## Requirements

- Python 3.10+
- Tesseract OCR
- OpenAI API key or HuggingFace models
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Tanvisharma31/mini-judge-Claw-LegalTech-Assessment.git
   cd minijudge
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

4. Set up environment variables:
   - Copy `template.env` to `.env`
   - Add your OpenAI API key or configure for HuggingFace
   - Set the path to Tesseract if it's not in your system PATH

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Upload a court judgment document (PDF or image)

4. Process the document by clicking the "Process Document" button

5. Once processing is complete, ask questions about the judgment in the text input field

6. View the AI-generated answer and the source text chunks that inform the answer

## Example Questions

- "Who were the parties in this case?"
- "What was the main legal issue in dispute?"
- "What precedents were cited in this judgment?"
- "What was the court's ruling on [specific issue]?"
- "What remedies were granted to the plaintiff?"

## Limitations

- OCR accuracy depends on document quality and formatting
- Processing large documents may take significant time
- Answers are based on the content available in the document
- Not a substitute for professional legal advice

## Customization

You can customize various aspects of MiniJudge:

- **Chunk Size and Overlap**: Adjust these parameters in the sidebar to optimize for your documents
- **LLM Provider**: Configure to use either OpenAI or HuggingFace models
- **OCR Language**: Change the language for OCR processing in the `.env` file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

MiniJudge is a prototype tool designed for research and educational purposes. It is not a substitute for legal advice, and its outputs should not be considered legal opinions. Always consult with a qualified legal professional for legal matters.