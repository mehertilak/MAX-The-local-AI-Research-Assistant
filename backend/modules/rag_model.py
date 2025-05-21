import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Suppress unnecessary logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

class DocumentRAG:
    def __init__(self, persist_data: bool = False):
        """
        Initialize the DocumentRAG system.

        Args:
            persist_data (bool): If True, data will be persisted on disk. Otherwise, it will be stored temporarily.
        """
        self.persist_data = persist_data
        self.temp_dirs = []
        self.vector_store: Optional[Chroma] = None
        self.qa_chain = None

        # Initialize storage
        if self.persist_data:
            self.data_dir = Path("data")
            self.data_dir.mkdir(exist_ok=True)
            self.chroma_dir = self.data_dir / "chromadb"
        else:
            self.data_dir = TemporaryDirectory()
            self.chroma_dir = TemporaryDirectory()
            self.temp_dirs.extend([self.data_dir, self.chroma_dir])

        # AI components
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = OllamaLLM(
            model="llama3.2",
            temperature=0.7,  # More creative responses
        )
        logging.info("RAG model initialized")

    def load_document(self, file_path: str) -> Optional[str]:
        """
        Load and extract text from a PDF or TXT file.

        Args:
            file_path (str): Path to the document file (PDF or TXT).

        Returns:
            Optional[str]: Extracted text if successful, None otherwise.
        """
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            logging.error("Unsupported file format. Only PDF and TXT files are supported.")
            return None

        try:
            documents = loader.load()
            text = " ".join([doc.page_content for doc in documents])
            return text
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            return None

    def process_content(self, content: str, source: str) -> List[Dict]:
        """
        Process content into chunks with metadata.

        Args:
            content (str): The text content to process.
            source (str): The source of the content (e.g., file path).

        Returns:
            List[Dict]: A list of dictionaries containing text chunks and metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", "```", " ", ""],
            keep_separator=True
        )

        chunks = splitter.split_text(content)
        return [{
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_num": i,
                "hash": hashlib.md5(chunk.encode()).hexdigest(),
                "processed_at": datetime.now().isoformat()
            }
        } for i, chunk in enumerate(chunks)]

    def create_vector_store(self, documents: List[Dict]):
        """
        Create ChromaDB vector store from processed documents.

        Args:
            documents (List[Dict]): A list of dictionaries containing text chunks and metadata.
        """
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        self.vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embedder,
            persist_directory=self.chroma_dir.name if not self.persist_data else str(self.chroma_dir)
        )
        logging.info(f"Created vector store with {len(documents)} chunks")
        self.qa_chain = None # Reset the qa_chain

    def setup_qa_chain(self) -> RetrievalQA:
        """
        Create a RetrievalQA chain for question-answering.

        Returns:
            RetrievalQA: A configured QA chain.
        """
        if self.qa_chain:
            return self.qa_chain

        prompt_template = """You are an expert research assistant providing answers based ONLY on the provided context. Follow these guidelines:

1.  **Focus:** Directly answer the question with precise information from the context.  Avoid extraneous details or opinions.
2.  **Contextual Honesty:** If the answer is NOT found within the context, respond with: "I cannot answer based on the provided context." Do not try to make up an answer.
3.  **Synthesis & Structure:**  Combine information from multiple parts of the context if needed to provide a comprehensive answer. Organize your response logically:
    *   Begin with a concise, one-sentence summary.
    *   Follow with a well-structured explanation, using bullet points or numbered lists where appropriate to present steps, categories, or supporting details.
4.  **Source Attribution (If Possible):**  Whenever you use a specific fact or statement from the context, cite the source using the format:  (Source: \[Document Name], Chunk \[Chunk Number]) . If the original source is known include it after.
5. **Follow up:** Be polite and ask the user if there any more information or details about the answer they are interested in.

Context:
{context}

Question:
{question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)


        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        logging.info("Setup new QA Chain")
        return self.qa_chain


    def cleanup(self):
        """
        Clean up resources (e.g., vector store, temporary directories).
        """
        try:
            # Close ChromaDB connection
            if self.vector_store:
                self.vector_store.delete_collection()
                self.vector_store = None
                logging.info("ChromaDB collection closed")

            # Clean up temporary directories
            if hasattr(self, 'temp_dirs'):
                for temp_dir in self.temp_dirs:
                    try:
                        temp_dir.cleanup()
                        logging.info(f"Cleaned {temp_dir.name}")
                    except PermissionError:
                        logging.warning(f"Force cleaning {temp_dir.name}")
                        import shutil
                        shutil.rmtree(temp_dir.name, ignore_errors=True)
                    except Exception as e:
                        logging.error(f"Cleanup error: {str(e)}")

            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

def upload_file(file, upload_folder):
    """
    Saves the uploaded file to the specified directory.

    Args:
        file: The file object from Flask request.
        upload_folder (str): The path to the directory where the file should be saved.

    Returns:
         str: the path where the file was saved
         str: the name of the file
    """
    if file:
        filename = file.filename
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        logging.info(f"File saved to: {file_path}")
        return file_path, filename
    else:
      logging.error("File is None")
      return None, None

def process_rag_query(question, rag_model):
        """
        Process a RAG query.

        Args:
            question (str): The question to ask the model.
             rag_model (DocumentRAG): The rag model object
        Returns:
            dict: A dictionary containing the answer and source document from the model.
        """
        try:
            logging.info(f"Processing RAG query: {question}")
            qa_chain = rag_model.setup_qa_chain()
            if qa_chain is None:
                logging.error(f"Error setting up the qa_chain")
                return None

            response = qa_chain(question)

            return response
        except Exception as e:
            logging.error(f"Error in process_rag_query: {e}")
            return None
