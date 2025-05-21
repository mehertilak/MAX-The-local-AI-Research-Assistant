# File: web_crawler.py
import os
import hashlib
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Optional

import crawl4ai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import shutil


# Suppress unnecessary logs
import httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

class EphemeralRAG:
    def __init__(self, persist_data: bool = False):
        self.persist_data = persist_data
        self.vector_store: Optional[Chroma] = None
        self.qa_chain = None
        
        # Initialize storage
        if self.persist_data:
            self.data_dir = Path("data")
            self.data_dir.mkdir(exist_ok=True)
            self.chroma_dir = self.data_dir / "chromadb"
        else:
            self.data_dir = None
            self.chroma_dir = None

        # AI components
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = OllamaLLM(
            model="llama3.2", 
            temperature=0.7,  # More creative responses
              # Enable markdown formatting
        )
        
        # Crawler configuration
        self.crawler_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

    async def crawl_website(self, url: str) -> Optional[str]:
         """Crawl a website and return its markdown content"""
         self.temp_dirs = []
         if not self.persist_data:
            self.data_dir = TemporaryDirectory()
            self.chroma_dir = TemporaryDirectory()
            self.temp_dirs.extend([self.data_dir, self.chroma_dir])
         crawler = AsyncWebCrawler(config=self.crawler_config)
         await crawler.start()
         try:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(cache_mode=crawl4ai.CacheMode.BYPASS)
             )

            if result and result.success:
                 content = result.markdown_v2.raw_markdown
                 self._store_raw_content(url, content)
                 return content
            logging.error(f"Failed to crawl {url}")
            return None
         except Exception as e:
            logging.error(f"Crawling error: {str(e)}")
            return None
         finally:
             await crawler.close()

    def _store_raw_content(self, url: str, content: str):
        """Store raw markdown content"""
        storage_path = Path(self.data_dir.name if not self.persist_data else self.data_dir)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        raw_path = storage_path / "raw" / f"{url_hash}.md"
        raw_path.parent.mkdir(exist_ok=True)
        
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Stored content for {url}")

    def process_content(self, content: str, url: str) -> List[Dict]:
        """Process content into chunks with metadata"""
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
                "source": url,
                "chunk_num": i,
                "hash": hashlib.md5(chunk.encode()).hexdigest(),
                "processed_at": datetime.now().isoformat()
            }
        } for i, chunk in enumerate(chunks)]

    def create_vector_store(self, documents: List[Dict]):
        """Create ChromaDB vector store"""
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        self.vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embedder,
             persist_directory=self.chroma_dir.name if not self.persist_data else str(self.chroma_dir)
        )
        logging.info(f"Created vector store with {len(documents)} chunks")
        self.qa_chain = None


    def setup_qa_chain(self) -> RetrievalQA:
        """Create RAG question-answering chain"""
        if self.qa_chain:
            return self.qa_chain
        prompt_template = """Follow these rules strictly:
            1. Answer ONLY using the provided context below
            2. If the question cannot be answered with the context , check to see if any related topic is there and asnwer based on it  , if no reated topic is present say: 
            "I don't have enough information to answer this question."
            3. Start with a detailed explanation of the answer in a paragraph.
            4. If the answer contains multiple types, steps, or categories or anything that should be listed ,then list them as numbered points (e.g., 1), 2)). If the answer does not require numbered points, skip them and just give a detailed information and a summary .
            5. End with a concise summary of the answer that gives a proper ending.

            Context: {context}

            Question: {question}

            Answer:"""
    
        prompt = ChatPromptTemplate.from_template(prompt_template)
    
        self.qa_chain =  RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
            )
        logging.info("Setup new QA Chain")
        return self.qa_chain
    
    def cleanup(self):
        """Ensure complete resource cleanup"""
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
            if self.data_dir:
                 try:
                     shutil.rmtree(self.data_dir.name, ignore_errors=True)
                 except Exception as e:
                      logging.error(f"Error cleaning temporary dir {e}")


            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

async def main():
    # Initialize ephemeral RAG system
    rag = EphemeralRAG(persist_data=False)
    
    try:
        # Get URL from user
        url = input("Enter website URL to process: ").strip()
        if not url.startswith("http"):
            url = f"https://{url}"
            
        # Crawl and process
        content = await rag.crawl_website(url)
        if not content:
            print("Failed to crawl website")
            return
            
        documents = rag.process_content(content, url)
        rag.create_vector_store(documents)
        
        # Setup QA system
        qa_chain = rag.setup_qa_chain()
        
        # Interactive session
        while True:
            query = input("\nAsk a question (q to quit): ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                break
                
            result = qa_chain.invoke({"query": query})
            print("\n" + "=" * 50)
            print(result['result'])
            print("\n**Sources:**")
            for doc in result['source_documents'][:3]:
                print(f"- {doc.metadata['source']} (chunk {doc.metadata['chunk_num']})")
            print("=" * 50 + "\n")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    finally:
        rag.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
