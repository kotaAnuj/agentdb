import sqlite3
import openai
from datetime import datetime
import json
import re
from tqdm import tqdm
from openai import OpenAI
import backoff
from contextlib import contextmanager
from typing import List, Dict, Any
import logging
from pathlib import Path
import pandas as pd
import PyPDF2
from io import StringIO
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import streamlit as st
import fitz  # PyMuPDF

# Setup logging once
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Updated Constants
MAX_RETRIES = 3
MAX_CHUNK_SIZE = 500
DATABASE_PATH = "document_processing.db"
# Replace with your actual API key
API_KEY = "your-google-palm-api-key"  
BASE_URL = "https://generativelanguage.googleapis.com/v1/models"

# Additional constants
CHUNK_PROCESSING_THREADS = 4
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_FORMATS = {'.txt', '.csv', '.pdf'}

# Initialize OpenAI client with correct configuration
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def setup_database():
    """Initialize database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            metadata TEXT,
            total_chunks INTEGER,
            processed_chunks INTEGER,
            status TEXT,
            progress REAL,
            last_updated TEXT
        )
        """)

        # Create chunks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            chunk_index INTEGER,
            chunk_content TEXT,
            chunk_result TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        """)
        
        conn.commit()

def split_into_chunks(content: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split content into chunks with improved sentence handling"""
    # Use lookahead/lookbehind for better sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

@backoff.on_exception(
    backoff.expo,
    (openai.OpenAIError, Exception),
    max_tries=MAX_RETRIES
)
def process_chunk_with_gemini(chunk: str) -> str:
    """Process chunk with retry mechanism using backoff decorator"""
    try:
        response = client.chat.completions.create(
            model="gemini-pro",
            temperature=0.7,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant tasked with analyzing text and generating reasoning."},
                {"role": "user", "content": f"Analyze this text and generate CoT reasoning:\n{chunk}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        raise

def read_large_file_in_chunks(file_path: str, chunk_size: int = 8192) -> str:
    """Read large files in chunks to prevent memory issues"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            content.append(chunk)
    return ''.join(content)

def process_pdf(file_path: str) -> str:
    """Process PDF files efficiently"""
    content = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                content.append(page.extract_text())
        return ' '.join(content)
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def process_csv(file_path: str) -> str:
    """Process CSV files with chunking"""
    try:
        # Read CSV in chunks
        chunks = pd.read_csv(file_path, chunksize=1000)
        content = []
        for chunk in chunks:
            content.append(chunk.to_string(index=False))
        return '\n'.join(content)
    except Exception as e:
        logger.error(f"Error processing CSV {file_path}: {str(e)}")
        raise

def process_file_content(file_path: str) -> str:
    """Process different file types efficiently"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"Large file detected ({file_size} bytes). Processing in chunks...")
    
    try:
        if file_ext == '.pdf':
            return process_pdf(file_path)
        elif file_ext == '.csv':
            return process_csv(file_path)
        else:  # .txt files
            return read_large_file_in_chunks(file_path)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def parallel_chunk_processing(chunks: List[str]) -> List[Dict[str, str]]:
    """Process chunks in parallel using ThreadPoolExecutor"""
    results = []
    with ThreadPoolExecutor(max_workers=CHUNK_PROCESSING_THREADS) as executor:
        future_to_chunk = {
            executor.submit(process_chunk_with_gemini, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                results.append({
                    'index': chunk_index,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                raise

    # Sort results by index
    results.sort(key=lambda x: x['index'])
    return [r['result'] for r in results]

def process_document_from_file(
    file_path: str,
    title: str,
    author: str,
    date: str
) -> Dict[str, Any]:
    """Process document from file with improved handling for large files"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Copy file to temporary directory
            temp_file = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy2(file_path, temp_file)
            
            # Process file content
            content = process_file_content(temp_file)
            chunks = split_into_chunks(content)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Start transaction
                    conn.execute("BEGIN")
                    
                    total_chunks = len(chunks)
                    metadata = json.dumps({
                        "author": author,
                        "date": date,
                        "original_file": file_path
                    })
                    
                    # Insert document
                    cursor.execute("""
                        INSERT INTO documents (title, metadata, total_chunks, processed_chunks, status, progress, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (title, metadata, total_chunks, 0, "in_progress", 0.0, datetime.now().isoformat()))
                    document_id = cursor.lastrowid
                    
                    # Process chunks in parallel
                    chunk_results = parallel_chunk_processing(chunks)
                    
                    # Save results
                    for i, (chunk, result) in enumerate(zip(chunks, chunk_results)):
                        cursor.execute("""
                            INSERT INTO chunks (document_id, chunk_index, chunk_content, chunk_result)
                            VALUES (?, ?, ?, ?)
                        """, (document_id, i, chunk, result))
                        
                        # Update progress
                        progress = (i + 1) / total_chunks
                        cursor.execute("""
                            UPDATE documents
                            SET processed_chunks = ?, progress = ?, last_updated = ?
                            WHERE id = ?
                        """, (i + 1, progress, datetime.now().isoformat(), document_id))
                        conn.commit()
                    
                    # Mark as completed
                    cursor.execute("""
                        UPDATE documents
                        SET status = ?, last_updated = ?
                        WHERE id = ?
                    """, ("completed", datetime.now().isoformat(), document_id))
                    
                    conn.commit()
                    
                    return {
                        "id": document_id,
                        "title": title,
                        "metadata": {
                            "author": author,
                            "date": date,
                            "original_file": file_path
                        },
                        "total_chunks": total_chunks,
                        "processed_chunks": total_chunks,
                        "status": "completed",
                        "progress": 1.0,
                        "last_updated": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error processing document: {str(e)}")
                    raise
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

def get_document_status(document_id: int) -> Dict[str, Any]:
    """Retrieve the current status of a document"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT title, metadata, total_chunks, processed_chunks, status, progress, last_updated
            FROM documents
            WHERE id = ?
        """, (document_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                "id": document_id,
                "title": result[0],
                "metadata": json.loads(result[1]),
                "total_chunks": result[2],
                "processed_chunks": result[3],
                "status": result[4],
                "progress": result[5],
                "last_updated": result[6]
            }
        return None

def get_document_results(document_id: int) -> List[Dict[str, str]]:
    """Retrieve all processed chunks and their results for a document"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chunk_index, chunk_content, chunk_result
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
        """, (document_id,))
        
        results = cursor.fetchall()
        return [
            {
                "chunk_index": row[0],
                "content": row[1],
                "result": row[2]
            }
            for row in results
        ]

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        api_key = os.getenv("GOOGLE_PALM_API_KEY", "your-api-key-here")  # Use environment variable
        st.session_state.agent = DocumentAgent(api_key)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

def get_file_extension(uploaded_file) -> str:
    """Get clean file extension from uploaded file"""
    # Get original filename
    filename = uploaded_file.name
    # Extract extension and clean it
    extension = os.path.splitext(filename)[1].lower()
    # Remove the dot and return
    return extension.replace('.', '')

def file_uploader_section():
    """Handle file upload and processing"""
    supported_types = ["pdf", "txt", "csv"]
    uploaded_file = st.file_uploader("Upload Document", type=supported_types)
    
    if uploaded_file:
        try:
            # Get proper file extension
            file_extension = get_file_extension(uploaded_file)
            
            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                # Write file content
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name  # Store the path
                
            # File inputs
            metadata = {
                "title": st.text_input("Document Title", uploaded_file.name),
                "author": st.text_input("Author", "Unknown"),
                "date": st.date_input("Document Date").isoformat()
            }
            
            analysis_types = st.multiselect(
                "Select Analysis Types",
                ["summary", "insights", "questions", "actions", "critique"],
                default=["summary"]
            )
            
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        # Process the document
                        doc_id = st.session_state.agent.process_file(tmp_file_path, metadata)
                        st.session_state.current_doc_id = doc_id
                        
                        # Perform selected analyses
                        results = {}
                        for analysis_type in analysis_types:
                            with st.spinner(f"Generating {analysis_type}..."):
                                result = st.session_state.agent.analyze_document(doc_id, analysis_type)
                                results[analysis_type] = result
                        
                        st.session_state.analysis_results = results
                        st.success("Document processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}", exc_info=True)
                    finally:
                        # Cleanup temporary file
                        try:
                            os.unlink(tmp_file_path)
                        except Exception as e:
                            logger.error(f"Error cleaning up temporary file: {str(e)}")

        except Exception as e:
            st.error(f"Error handling file upload: {str(e)}")
            logger.error(f"File upload error: {str(e)}", exc_info=True)

def display_analysis_results():
    """Display document analysis results"""
    if st.session_state.analysis_results:
        st.header("Analysis Results")
        
        for analysis_type, result in st.session_state.analysis_results.items():
            with st.expander(f"{analysis_type.title()} Analysis"):
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(result["result"])
                    st.caption(f"Generated at: {result['timestamp']}")

def chat_interface():
    """Display chat interface"""
    st.header("Document Chat")
    
    if st.session_state.current_doc_id:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the document"):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat_query(
                    st.session_state.current_doc_id, 
                    prompt
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            
            st.rerun()
    else:
        st.info("Upload and process a document to start chatting")

def main():
    st.set_page_config(page_title="Document AI Agent", layout="wide")
    initialize_session_state()
    
    st.title("Document AI Agent")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        file_uploader_section()
        display_analysis_results()
    
    with col2:
        chat_interface()

if __name__ == "__main__":
    main()
    
    
    





from datetime import datetime
import json
import sqlite3
import tempfile
from typing import Any, Dict, List
import logging
from pathlib import Path
import streamlit as st
from openai import OpenAI
import fitz
from concurrent.futures import ThreadPoolExecutor

# Import our document processing system


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )
        setup_database()
        
    def process_file(self, file_path: str, metadata: Dict[str, Any]) -> int:
        """Process any supported file using existing system"""
        try:
            result = process_document_from_file(
                file_path=file_path,
                title=metadata.get('title', 'Untitled'),
                author=metadata.get('author', 'Unknown'),
                date=metadata.get('date', datetime.now().isoformat())
            )
            return result["id"]
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def get_document_context(self, document_id: int) -> str:
        """Get processed document context"""
        results = get_document_results(document_id)
        return "\n".join([
            f"Section {r['chunk_index']}:\n{r['content']}\nAnalysis: {r['result']}"
            for r in results
        ])

    def analyze_document(self, document_id: int, analysis_type: str) -> Dict[str, Any]:
        """Perform specific type of analysis on document"""
        context = self.get_document_context(document_id)
        
        prompts = {
            "summary": "Provide a comprehensive summary of the document.",
            "insights": "Extract key insights and findings from the document.",
            "questions": "Generate important questions and answers about the document content.",
            "actions": "Recommend specific actions based on the document content.",
            "critique": "Provide a critical analysis of the document's content and arguments."
        }

        try:
            response = self.client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {"role": "system", "content": prompts.get(analysis_type, "Analyze the document content.")},
                    {"role": "user", "content": f"Document content and analysis:\n{context}"}
                ]
            )
            
            return {
                "type": analysis_type,
                "result": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {"error": str(e)}

    def chat_query(self, document_id: int, query: str) -> str:
        """Handle chat queries about the document"""
        context = self.get_document_context(document_id)
        
        try:
            response = self.client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant answering questions about a document."},
                    {"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {query}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat query: {str(e)}")
            return f"Error processing query: {str(e)}"    
