#!/usr/bin/env python3
"""
BladderCancerAgent - 방광암 EAU 가이드라인 AI Agent 핵심 클래스
Ollama Qwen 모델과 RAG 기능을 통합한 독립 에이전트
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

# 필수 라이브러리
import ollama
import torch
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from tqdm import tqdm
import psutil

class BladderCancerAgent:
    """방광암 EAU 가이드라인 AI Agent 핵심 클래스"""
    
    def __init__(self, config):
        """
        에이전트 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 상태 변수
        self.is_initialized = False
        self.ollama_client = None
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.pdf_loaded = False
        
        # 캐시 디렉토리 생성
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # 인코딩 설정
        self._setup_encoding()
        
        # 한국어-영어 의료용어 매핑
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
        
        # 방광암 관련 키워드 (기존 MCP 서버에서 가져옴)
        self.bladder_keywords = [
            "BCG", "TURBT", "NMIBC", "MIBC", "bladder", "cancer", "urothelial", "carcinoma",
            "cystoscopy", "resection", "chemotherapy", "immunotherapy", "recurrence",
            "survival", "prognosis", "stage", "grade", "risk", "treatment", "therapy",
            "guidelines", "recommendation", "EAU", "urinary", "tract", "hematuria",
            "frequency", "urgency", "incontinence", "biopsy", "histopathology",
            "invasion", "metastasis", "epithelium", "malignant", "benign"
        ]

    def _setup_encoding(self):
        """인코딩 설정"""
        try:
            import locale
            # 로케일 설정
            if os.name == 'nt':  # Windows
                locale.setlocale(locale.LC_ALL, 'Korean_Korea.utf8')
            else:  # Linux/macOS
                locale.setlocale(locale.LC_ALL, 'ko_KR.utf8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.utf8')
            except:
                pass
        
        # 환경 변수 설정
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    def initialize(self) -> bool:
        """
        에이전트 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("에이전트 초기화 시작")
            
            # 1. Ollama 클라이언트 초기화
            if not self._init_ollama():
                return False
            
            # 2. 임베딩 모델 초기화
            if not self._init_embedding_model():
                return False
            
            # 3. ChromaDB 초기화
            if not self._init_chromadb():
                return False
            
            # 4. PDF 로드 및 벡터화
            if not self._load_pdf_and_vectorize():
                return False
            
            self.is_initialized = True
            self.logger.info("에이전트 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"에이전트 초기화 실패: {str(e)}")
            return False

    def _init_ollama(self) -> bool:
        """Ollama 클라이언트 초기화"""
        try:
            self.logger.info("Ollama 클라이언트 초기화 중...")
            
            # Ollama 클라이언트 생성
            self.ollama_client = ollama.Client(host=self.config.ollama_host)
            
            # 모델 존재 확인
            models = self.ollama_client.list()
            self.logger.info(f"Ollama 응답: {models}")
            
            # 안전한 모델 목록 추출
            if 'models' in models and isinstance(models['models'], list):
                model_names = []
                for model in models['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
                    else:
                        self.logger.warning(f"예상하지 못한 모델 형식: {model}")
            else:
                self.logger.warning(f"예상하지 못한 Ollama 응답 형식: {models}")
                model_names = []
            
            if self.config.model_name not in model_names:
                self.logger.info(f"모델 '{self.config.model_name}'이 설치되지 않음")
                self.logger.info(f"설치된 모델: {model_names}")
                
                # 모델 자동 다운로드
                if not self._download_model():
                    return False
            
            # 모델 테스트
            response = self.ollama_client.generate(
                model=self.config.model_name,
                prompt="Hello",
                stream=False
            )
            
            if not response.get('response'):
                self.logger.error("Ollama 모델 응답 테스트 실패")
                return False
            
            self.logger.info(f"Ollama 모델 '{self.config.model_name}' 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"Ollama 초기화 실패: {str(e)}")
            return False

    def _download_model(self) -> bool:
        """Ollama 모델 다운로드"""
        try:
            print(f"🔄 모델 '{self.config.model_name}' 다운로드 중... (약 400MB)")
            print("이 작업은 몇 분 정도 소요될 수 있습니다.")
            
            # Ollama pull 명령 실행
            import subprocess
            import sys
            
            result = subprocess.run(
                ['ollama', 'pull', self.config.model_name], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30분 타임아웃
            )
            
            if result.returncode == 0:
                print(f"✅ 모델 '{self.config.model_name}' 다운로드 완료")
                return True
            else:
                print(f"❌ 모델 다운로드 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ 모델 다운로드 시간 초과")
            return False
        except Exception as e:
            print(f"❌ 모델 다운로드 중 오류: {str(e)}")
            return False

    def _init_embedding_model(self) -> bool:
        """임베딩 모델 초기화"""
        try:
            self.logger.info("임베딩 모델 초기화 중...")
            
            # GPU 사용 가능 여부 확인
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"사용 디바이스: {device}")
            
            # 다국어 임베딩 모델 로드
            model_name = self.config.embedding_model
            self.embedding_model = SentenceTransformer(model_name, device=device)
            
            # 모델 테스트
            test_text = "BCG 치료의 부작용은 무엇인가요?"
            embedding = self.embedding_model.encode([test_text])
            
            if embedding is None or len(embedding) == 0:
                self.logger.error("임베딩 모델 테스트 실패")
                return False
            
            self.logger.info(f"임베딩 모델 '{model_name}' 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"임베딩 모델 초기화 실패: {str(e)}")
            return False

    def _init_chromadb(self) -> bool:
        """ChromaDB 초기화"""
        try:
            self.logger.info("ChromaDB 초기화 중...")
            
            # ChromaDB 클라이언트 생성
            self.chroma_client = chromadb.PersistentClient(
                path=str(Path(self.config.cache_dir) / "chroma_db")
            )
            
            # 컬렉션 생성 또는 가져오기
            collection_name = "bladder_cancer_guidelines"
            
            # 기존 컬렉션이 있으면 삭제 (재생성)
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            
            # 새 컬렉션 생성
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            
            self.logger.info("ChromaDB 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"ChromaDB 초기화 실패: {str(e)}")
            return False

    def _load_pdf_and_vectorize(self) -> bool:
        """PDF 로드 및 벡터화"""
        try:
            self.logger.info("PDF 로드 및 벡터화 시작...")
            
            # PDF 파일 경로
            pdf_path = Path(self.config.pdf_path)
            if not pdf_path.exists():
                self.logger.error(f"PDF 파일이 없습니다: {pdf_path}")
                return False
            
            # PDF 텍스트 추출
            documents = self._extract_pdf_text(pdf_path)
            if not documents:
                self.logger.error("PDF에서 텍스트 추출 실패")
                return False
            
            # 방광암 관련 문서만 필터링
            filtered_docs = self._filter_relevant_documents(documents)
            if not filtered_docs:
                self.logger.error("방광암 관련 문서가 없습니다")
                return False
            
            # 벡터화 및 ChromaDB에 저장
            self._vectorize_and_store(filtered_docs)
            
            self.pdf_loaded = True
            self.logger.info(f"PDF 벡터화 완료: {len(filtered_docs)}개 문서")
            return True
            
        except Exception as e:
            self.logger.error(f"PDF 로드 및 벡터화 실패: {str(e)}")
            return False

    def _extract_pdf_text(self, pdf_path: Path) -> List[str]:
        """PDF에서 텍스트 추출"""
        try:
            documents = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # 텍스트 정리
                        cleaned_text = self._clean_text(text)
                        if cleaned_text:
                            documents.append(cleaned_text)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 실패: {str(e)}")
            return []

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 유니코드 정규화
        text = unicodedata.normalize('NFC', text)
        
        # 불필요한 문자 제거
        text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\{\}\'\"\/\%\&\+\=\<\>\!\?\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text

    def _filter_relevant_documents(self, documents: List[str]) -> List[str]:
        """방광암 관련 문서 필터링"""
        filtered = []
        
        for doc in documents:
            # 키워드 매칭
            doc_lower = doc.lower()
            if any(keyword.lower() in doc_lower for keyword in self.bladder_keywords):
                # 최소 길이 확인
                if len(doc.split()) >= 10:
                    filtered.append(doc)
        
        return filtered

    def _vectorize_and_store(self, documents: List[str]):
        """문서 벡터화 및 ChromaDB 저장"""
        try:
            # 배치 크기 설정
            batch_size = self.config.batch_size
            
            for i in tqdm(range(0, len(documents), batch_size), desc="문서 벡터화"):
                batch = documents[i:i+batch_size]
                
                # 임베딩 생성
                embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
                
                # ChromaDB에 저장
                ids = [f"doc_{i+j}" for j in range(len(batch))]
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch,
                    ids=ids
                )
                
        except Exception as e:
            self.logger.error(f"벡터화 및 저장 실패: {str(e)}")
            raise

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: 답변 결과
        """
        try:
            if not self.is_initialized:
                return {"success": False, "error": "에이전트가 초기화되지 않았습니다"}
            
            # 1. 한국어 질문 확장
            expanded_query = self._expand_korean_query(question)
            
            # 2. 관련 문서 검색
            relevant_docs = self._search_relevant_documents(expanded_query)
            
            if not relevant_docs:
                return {"success": False, "error": "관련 문서를 찾을 수 없습니다"}
            
            # 3. 컨텍스트 생성
            context = self._create_context(relevant_docs)
            
            # 4. AI 답변 생성
            answer = self._generate_answer(question, context)
            
            if not answer:
                return {"success": False, "error": "답변 생성 실패"}
            
            # 5. 결과 반환
            return {
                "success": True,
                "answer": answer,
                "sources": [doc["document"] for doc in relevant_docs],
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"질문 처리 실패: {str(e)}")
            return {"success": False, "error": str(e)}

    def _expand_korean_query(self, query: str) -> str:
        """한국어 질문을 영어로 확장"""
        expanded = query
        
        for korean, english in self.korean_to_english.items():
            if korean in query:
                expanded += f" {english}"
        
        return expanded

    def _search_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])
            
            # ChromaDB에서 검색
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=self.config.top_k
            )
            
            # 결과 정리
            relevant_docs = []
            for i, doc in enumerate(results['documents'][0]):
                relevant_docs.append({
                    "document": doc,
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                })
            
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"문서 검색 실패: {str(e)}")
            return []

    def _create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """컨텍스트 생성"""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[문서 {i}]\n{doc['document']}\n")
        
        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """AI 답변 생성"""
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(question, context)
            
            # Ollama 모델 호출
            response = self.ollama_client.generate(
                model=self.config.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k_generate,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens
                }
            )
            
            answer = response.get('response', '').strip()
            
            if not answer:
                return "죄송합니다. 답변을 생성할 수 없습니다."
            
            return answer
            
        except Exception as e:
            self.logger.error(f"답변 생성 실패: {str(e)}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    def _create_prompt(self, question: str, context: str) -> str:
        """프롬프트 생성"""
        prompt = f"""당신은 방광암 치료 전문가입니다. EAU(European Association of Urology) 가이드라인을 기반으로 정확하고 신뢰할 수 있는 의료 정보를 제공합니다.

아래 제공된 의료 문서를 참고하여 질문에 답변해주세요.

【참고 문서】
{context}

【질문】
{question}

【답변 지침】
1. 제공된 문서의 내용만을 바탕으로 답변하세요.
2. 의학적으로 정확하고 신뢰할 수 있는 정보만 제공하세요.
3. 불확실한 내용은 "가이드라인에 명시되지 않았습니다"라고 답변하세요.
4. 구체적인 치료법이나 약물 정보는 전문의와 상담하도록 안내하세요.
5. 한국어로 친절하고 이해하기 쉽게 답변하세요.

【답변】
"""
        return prompt

    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보"""
        try:
            status = {
                "initialized": self.is_initialized,
                "ollama_connected": False,
                "model_available": False,
                "pdf_loaded": self.pdf_loaded,
                "vectordb_ready": False
            }
            
            # Ollama 연결 상태 확인
            if self.ollama_client:
                try:
                    models = self.ollama_client.list()
                    status["ollama_connected"] = True
                    
                    model_names = [model['name'] for model in models['models']]
                    status["model_available"] = self.config.model_name in model_names
                except:
                    pass
            
            # 벡터 DB 상태 확인
            if self.collection:
                try:
                    count = self.collection.count()
                    status["vectordb_ready"] = count > 0
                    status["document_count"] = count
                except:
                    pass
            
            return status
            
        except Exception as e:
            self.logger.error(f"상태 확인 실패: {str(e)}")
            return {"error": str(e)}

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            
            if hasattr(self, 'chroma_client'):
                del self.chroma_client
            
            if hasattr(self, 'collection'):
                del self.collection
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {str(e)}")