#!/usr/bin/env python3
"""
BladderCancerAgent - Bladder Cancer EAU Guidelines AI Agent Core Class
Independent agent integrating Ollama Qwen model with RAG functionality
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import unicodedata
import re
import time

# Required libraries
import ollama
import torch
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from tqdm import tqdm
import psutil

class BladderCancerAgent:
    """Bladder Cancer EAU Guidelines AI Agent Core Class"""
    
    def __init__(self, config):
        """
        Initialize agent
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State variables
        self.is_initialized = False
        self.ollama_client = None
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.pdf_loaded = False
        
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Setup encoding
        self._setup_encoding()
        
        # Korean-English medical term mapping
        self.korean_to_english = {
            "방광암": "bladder cancer urothelial carcinoma",
            "비근침윤성": "non-muscle invasive NMIBC",
            "근침윤성": "muscle invasive MIBC",
            "BCG": "BCG bacillus calmette guerin",
            "방광경": "cystoscopy",
            "경요도방광종양절제술": "TURBT transurethral resection",
            "화학요법": "chemotherapy",
            "면역요법": "immunotherapy",
            "재발": "recurrence",
            "부작용": "side effects adverse",
            "치료": "treatment therapy",
            "수술": "surgery operation",
            "진단": "diagnosis",
            "예후": "prognosis",
            "생존율": "survival rate",
            "병기": "stage staging",
            "등급": "grade grading",
            "위험도": "risk factors",
            "가이드라인": "guidelines recommendations",
            "표준치료": "standard treatment",
            "항암": "anticancer chemotherapy",
            "방사선": "radiation radiotherapy",
            "절제": "resection removal",
            "생검": "biopsy",
            "조직검사": "histopathology",
            "종양": "tumor neoplasm",
            "악성": "malignant",
            "양성": "benign",
            "전이": "metastasis",
            "침윤": "invasion",
            "상피": "epithelium urothelial",
            "요로": "urinary tract",
            "배뇨": "urination voiding",
            "혈뇨": "hematuria",
            "빈뇨": "frequency",
            "요절박": "urgency",
            "요실금": "incontinence"
        }
        
        # Bladder cancer related keywords
        self.bladder_keywords = [
            "BCG", "TURBT", "NMIBC", "MIBC", "bladder", "cancer", "urothelial", "carcinoma",
            "cystoscopy", "resection", "chemotherapy", "immunotherapy", "recurrence",
            "survival", "prognosis", "stage", "grade", "risk", "treatment", "therapy",
            "guidelines", "recommendation", "EAU", "urinary", "tract", "hematuria",
            "frequency", "urgency", "incontinence", "biopsy", "histopathology",
            "invasion", "metastasis", "epithelium", "malignant", "benign"
        ]

    def _setup_encoding(self):
        """Setup encoding"""
        try:
            import locale
            # Set locale
            if os.name == 'nt':  # Windows
                try:
                    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
                except:
                    pass
            else:  # Linux/macOS
                try:
                    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
                except:
                    pass
        except:
            pass
        
        # Set environment variables
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    def initialize(self) -> bool:
        """
        Initialize agent
        
        Returns:
            bool: Initialization success status
        """
        try:
            self.logger.info("Starting agent initialization")
            
            # 1. Initialize Ollama client
            if not self._init_ollama():
                return False
            
            # 2. Initialize embedding model
            if not self._init_embedding_model():
                return False
            
            # 3. Initialize ChromaDB
            if not self._init_chromadb():
                return False
            
            # 4. Load PDF and vectorize
            if not self._load_pdf_and_vectorize():
                return False
            
            self.is_initialized = True
            self.logger.info("Agent initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {str(e)}")
            return False

    def _init_ollama(self) -> bool:
        """Initialize Ollama client"""
        try:
            self.logger.info("Initializing Ollama client...")
            
            # Create Ollama client
            self.ollama_client = ollama.Client(host=self.config.ollama_host)
            
            # Check model existence
            models = self.ollama_client.list()
            self.logger.info(f"Ollama response: {models}")
            
            # Safe model list extraction
            if 'models' in models and isinstance(models['models'], list):
                model_names = []
                for model in models['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
                    else:
                        self.logger.warning(f"Unexpected model format: {model}")
            else:
                self.logger.warning(f"Unexpected Ollama response format: {models}")
                model_names = []
            
            if self.config.model_name not in model_names:
                self.logger.info(f"Model '{self.config.model_name}' not installed")
                self.logger.info(f"Installed models: {model_names}")
                
                # Auto download model
                if not self._download_model():
                    return False
            
            # Test model
            response = self.ollama_client.generate(
                model=self.config.model_name,
                prompt="Hello",
                stream=False
            )
            
            if not response.get('response'):
                self.logger.error("Ollama model response test failed")
                return False
            
            self.logger.info(f"Ollama model '{self.config.model_name}' initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Ollama initialization failed: {str(e)}")
            return False

    def _download_model(self) -> bool:
        """Download Ollama model"""
        try:
            print(f"🔄 Downloading model '{self.config.model_name}'... (about 400MB)")
            print("This may take a few minutes.")
            
            # Execute Ollama pull command
            import subprocess
            import sys
            
            result = subprocess.run(
                ['ollama', 'pull', self.config.model_name], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Model '{self.config.model_name}' download completed")
                return True
            else:
                print(f"❌ Model download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Model download timeout")
            return False
        except Exception as e:
            print(f"❌ Error during model download: {str(e)}")
            return False

    def _init_embedding_model(self) -> bool:
        """Initialize embedding model"""
        try:
            self.logger.info("Initializing embedding model...")
            
            # Check GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            # Load multilingual embedding model
            model_name = self.config.embedding_model
            self.embedding_model = SentenceTransformer(model_name, device=device)
            
            # Test model
            test_text = "What are the side effects of BCG therapy?"
            embedding = self.embedding_model.encode([test_text])
            
            if embedding is None or len(embedding) == 0:
                self.logger.error("Embedding model test failed")
                return False
            
            self.logger.info(f"Embedding model '{model_name}' initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Embedding model initialization failed: {str(e)}")
            return False

    def _init_chromadb(self) -> bool:
        """Initialize ChromaDB"""
        try:
            self.logger.info("Initializing ChromaDB...")
            
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(Path(self.config.cache_dir) / "chroma_db")
            )
            
            # Create or get collection
            collection_name = "bladder_cancer_guidelines"
            
            # Delete existing collection if exists (recreate)
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            
            self.logger.info("ChromaDB initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"ChromaDB initialization failed: {str(e)}")
            return False

    def _load_pdf_and_vectorize(self) -> bool:
        """Load PDF and vectorize"""
        try:
            self.logger.info("Starting PDF loading and vectorization...")
            
            # PDF file path
            pdf_path = Path(self.config.pdf_path)
            if not pdf_path.exists():
                self.logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Extract PDF text
            documents = self._extract_pdf_text(pdf_path)
            if not documents:
                self.logger.error("Failed to extract text from PDF")
                return False
            
            # Filter bladder cancer related documents only
            filtered_docs = self._filter_relevant_documents(documents)
            if not filtered_docs:
                self.logger.error("No bladder cancer related documents found")
                return False
            
            # Vectorize and store in ChromaDB
            self._vectorize_and_store(filtered_docs)
            
            self.pdf_loaded = True
            self.logger.info(f"PDF vectorization completed: {len(filtered_docs)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"PDF loading and vectorization failed: {str(e)}")
            return False

    def _extract_pdf_text(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF"""
        try:
            documents = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text and text.strip():
                            # Safe text processing
                            try:
                                # Clean text
                                cleaned_text = self._clean_text(text)
                                if cleaned_text and len(cleaned_text.strip()) > 10:
                                    documents.append(cleaned_text)
                                    
                            except UnicodeError as e:
                                self.logger.warning(f"Page {page_num} encoding error, skipping: {e}")
                                continue
                                
                    except Exception as e:
                        self.logger.warning(f"Page {page_num} processing failed, skipping: {e}")
                        continue
            
            return documents
            
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {str(e)}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean text"""
        if not text:
            return ""
        
        try:
            # Safe encoding handling
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Unicode normalization
            text = unicodedata.normalize('NFC', text)
            
            # Remove unnecessary characters (safer pattern)
            # Keep only English, numbers, Korean, basic punctuation
            text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\{\}\'\"\/\%\&\+\=\<\>\!\?]', ' ', text, flags=re.UNICODE)
            
            # Remove consecutive spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing spaces
            text = text.strip()
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Error during text cleaning: {e}")
            # Return safe version on error
            try:
                return str(text).encode('ascii', errors='ignore').decode('ascii')
            except:
                return ""

    def _filter_relevant_documents(self, documents: List[str]) -> List[str]:
        """Filter bladder cancer related documents"""
        filtered = []
        
        for doc in documents:
            # Keyword matching
            doc_lower = doc.lower()
            if any(keyword.lower() in doc_lower for keyword in self.bladder_keywords):
                # Check minimum length
                if len(doc.split()) >= 10:
                    filtered.append(doc)
        
        return filtered

    def _vectorize_and_store(self, documents: List[str]):
        """Vectorize documents and store in ChromaDB"""
        try:
            # Set batch size
            batch_size = self.config.batch_size
            
            for i in tqdm(range(0, len(documents), batch_size), desc="Document vectorization"):
                batch = documents[i:i+batch_size]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
                
                # Generate IDs
                ids = [f"doc_{i+j}" for j in range(len(batch))]
                
                # Store in ChromaDB
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch,
                    ids=ids
                )
                
        except Exception as e:
            self.logger.error(f"Vectorization and storage failed: {str(e)}")
            raise

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Generate answer to question
        
        Args:
            question: User question
            
        Returns:
            Dict: Answer result
        """
        try:
            if not self.is_initialized:
                return {"success": False, "error": "Agent not initialized"}
            
            # 1. Expand Korean query
            expanded_query = self._expand_korean_query(question)
            
            # 2. Search relevant documents
            relevant_docs = self._search_relevant_documents(expanded_query)
            
            if not relevant_docs:
                return {"success": False, "error": "No relevant documents found"}
            
            # 3. Create context
            context = self._create_context(relevant_docs)
            
            # 4. Generate AI answer
            answer = self._generate_answer(question, context)
            
            if not answer:
                return {"success": False, "error": "Answer generation failed"}
            
            # 5. Return result
            return {
                "success": True,
                "answer": answer,
                "sources": [doc["document"] for doc in relevant_docs],
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Question processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _expand_korean_query(self, query: str) -> str:
        """Expand Korean question to English"""
        expanded = query
        
        for korean, english in self.korean_to_english.items():
            if korean in query:
                expanded += f" {english}"
        
        return expanded

    def _search_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.config.top_k
            )
            
            # Organize results
            relevant_docs = []
            for i, doc in enumerate(results['documents'][0]):
                relevant_docs.append({
                    "document": doc,
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                })
            
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Document search failed: {str(e)}")
            return []

    def _create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Create context"""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[Document {i}]\\n{doc['document']}\\n")
        
        return "\\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate AI answer"""
        try:
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Call Ollama model
            response = self.ollama_client.generate(
                model=self.config.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            if response and response.get('response'):
                return response['response'].strip()
            else:
                self.logger.error("Empty response from Ollama")
                return ""
                
        except Exception as e:
            self.logger.error(f"Answer generation failed: {str(e)}")
            return ""

    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for AI model"""
        prompt = f"""You are a medical AI assistant specializing in bladder cancer based on EAU (European Association of Urology) guidelines.

Context from EAU Guidelines:
{context}

Question: {question}

Please provide a comprehensive answer based on the provided EAU guidelines context. If the information is not available in the context, clearly state that. Answer in the same language as the question was asked.

Answer:"""
        
        return prompt

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            status = {
                "ollama_connected": False,
                "model_available": False,
                "pdf_loaded": self.pdf_loaded,
                "vectordb_ready": False
            }
            
            # Check Ollama connection
            try:
                if self.ollama_client:
                    models = self.ollama_client.list()
                    status["ollama_connected"] = True
                    
                    # Check model availability
                    if 'models' in models:
                        model_names = [m.get('name', '') for m in models['models']]
                        status["model_available"] = self.config.model_name in model_names
            except:
                pass
            
            # Check ChromaDB
            try:
                if self.collection:
                    count = self.collection.count()
                    status["vectordb_ready"] = count > 0
            except:
                pass
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {str(e)}")
            return {}

if __name__ == "__main__":
    # Simple test
    from config import Config
    
    config = Config()
    agent = BladderCancerAgent(config)
    
    if agent.initialize():
        print("Agent initialized successfully")
        
        # Test question
        result = agent.ask_question("What are the side effects of BCG therapy?")
        if result['success']:
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")
    else:
        print("Agent initialization failed")