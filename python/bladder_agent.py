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
            import sys
            
            # 로케일 설정
            if os.name == 'nt':  # Windows
                try:
                    locale.setlocale(locale.LC_ALL, 'Korean_Korea.utf8')
                except:
                    try:
                        locale.setlocale(locale.LC_ALL, 'ko_KR.utf8')
                    except:
                        locale.setlocale(locale.LC_ALL, 'en_US.utf8')
            else:  # Linux/macOS
                try:
                    locale.setlocale(locale.LC_ALL, 'ko_KR.utf8')
                except:
                    try:
                        locale.setlocale(locale.LC_ALL, 'en_US.utf8')
                    except:
                        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except Exception as e:
            self.logger.warning(f"로케일 설정 실패: {e}")
        
        # 환경 변수 설정
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        
        # Python 스트림 인코딩 설정
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
        if hasattr(sys.stdin, 'reconfigure'):
            sys.stdin.reconfigure(encoding='utf-8')

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
            print(f"모델 '{self.config.model_name}' 다운로드 중... (약 1GB)")
            print("이 작업은 몇 분 정도 소요될 수 있습니다.")
            print("-" * 60)
            
            # Ollama pull 명령 실행
            import subprocess
            import sys
            import re
            
            # 실시간 출력을 위한 Popen 사용
            process = subprocess.Popen(
                ['ollama', 'pull', self.config.model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 진행상황 추적 변수
            current_step = "준비 중"
            progress_count = 0
            last_significant_line = ""
            
            # 단순한 진행 표시기
            with tqdm(desc=current_step, bar_format='{desc}: {n}/100 {bar}', total=100) as pbar:
                for line in iter(process.stdout.readline, ''):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 중요한 라인만 출력
                    is_important = False
                    
                    # pulling manifest, config, model 등 주요 단계 감지
                    if 'pulling manifest' in line.lower():
                        current_step = "Downloading manifest"
                        pbar.set_description(current_step)
                        pbar.update(10)
                        is_important = True
                    elif 'pulling' in line.lower() and 'sha256:' in line.lower():
                        if 'config' in line.lower():
                            current_step = "Downloading config"
                        else:
                            current_step = "Downloading model layers"
                        pbar.set_description(current_step)
                        progress_count += 15
                        pbar.update(15)
                        is_important = True
                    elif 'verifying' in line.lower():
                        current_step = "Verifying download"
                        pbar.set_description(current_step)
                        pbar.update(10)
                        is_important = True
                    elif 'success' in line.lower() or 'complete' in line.lower():
                        current_step = "Download complete"
                        pbar.set_description(current_step)
                        pbar.update(100 - pbar.n)  # 나머지 모두 채우기
                        is_important = True
                    
                    # 중요한 라인만 출력 (중복 방지)
                    if is_important and line != last_significant_line:
                        print(f"  {line}")
                        last_significant_line = line
            
            process.wait()
            print("-" * 60)
            
            if process.returncode == 0:
                print(f"모델 '{self.config.model_name}' 다운로드 완료")
                return True
            else:
                print(f"모델 다운로드 실패 (종료 코드: {process.returncode})")
                return False
                
        except subprocess.TimeoutExpired:
            print("모델 다운로드 시간 초과")
            return False
        except Exception as e:
            print(f"모델 다운로드 중 오류: {str(e)}")
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
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text and text.strip():
                            # 안전한 텍스트 처리
                            try:
                                # UTF-8 인코딩 확인 및 변환
                                if isinstance(text, bytes):
                                    text = text.decode('utf-8', errors='ignore')
                                elif isinstance(text, str):
                                    # 문자열을 바이트로 변환 후 다시 디코딩하여 안전하게 처리
                                    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                                
                                # 텍스트 정리
                                cleaned_text = self._clean_text(text)
                                if cleaned_text and len(cleaned_text.strip()) > 10:
                                    documents.append(cleaned_text)
                                    
                            except UnicodeError as e:
                                self.logger.warning(f"페이지 {page_num} 인코딩 오류, 건너뜀: {e}")
                                continue
                                
                    except Exception as e:
                        self.logger.warning(f"페이지 {page_num} 처리 실패, 건너뜀: {e}")
                        continue
            
            return documents
            
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 실패: {str(e)}")
            return []

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        try:
            # 안전한 인코딩 처리
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # 유니코드 정규화
            text = unicodedata.normalize('NFC', text)
            
            # 불필요한 문자 제거 (더 안전한 패턴)
            # 영문, 숫자, 한글, 기본 구두점만 유지
            text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\{\}\'\"\/\%\&\+\=\<\>\!\?\uAC00-\uD7A3]', ' ', text, flags=re.UNICODE)
            
            # 연속된 공백 제거
            text = re.sub(r'\s+', ' ', text)
            
            # 앞뒤 공백 제거
            text = text.strip()
            
            return text
            
        except Exception as e:
            self.logger.warning(f"텍스트 정리 중 오류: {e}")
            # 오류 발생 시 원본 텍스트의 안전한 버전 반환
            try:
                return str(text).encode('ascii', errors='ignore').decode('ascii')
            except:
                return ""

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
            
            # 0. 질문 인코딩 안전 처리
            question = self._safe_encode_text(question)
            
            # 1. 한국어 질문 확장
            expanded_query = self._expand_korean_query(question)
            expanded_query = self._safe_encode_text(expanded_query)
            
            # 2. 관련 문서 검색
            relevant_docs = self._search_relevant_documents(expanded_query)
            
            if not relevant_docs:
                return {"success": False, "error": "관련 문서를 찾을 수 없습니다"}
            
            # 3. 컨텍스트 생성
            context = self._create_context(relevant_docs)
            context = self._safe_encode_text(context)
            
            # 4. AI 답변 생성
            answer = self._generate_answer(question, context)
            answer = self._safe_encode_text(answer) if answer else ""
            
            if not answer:
                return {"success": False, "error": "답변 생성 실패"}
            
            # 5. 결과 반환
            return {
                "success": True,
                "answer": answer,
                "sources": [self._create_source_summary(doc, i+1) for i, doc in enumerate(relevant_docs)],
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"질문 처리 실패: {str(e)}")
            return {"success": False, "error": str(e)}

    def _safe_encode_text(self, text: str) -> str:
        """텍스트 안전 인코딩 처리"""
        if not text:
            return ""
        
        try:
            # 이미 문자열인 경우
            if isinstance(text, str):
                # 유니코드 정규화 및 안전한 UTF-8 처리
                import unicodedata
                text = unicodedata.normalize('NFC', text)
                # 안전한 문자만 유지
                safe_text = ''.join(c for c in text if unicodedata.category(c) not in ['Cc', 'Cf', 'Cs', 'Co', 'Cn'])
                return safe_text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            
            # 바이트인 경우
            elif isinstance(text, bytes):
                # 여러 인코딩 방식을 시도
                for encoding in ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']:
                    try:
                        decoded = text.decode(encoding)
                        return self._safe_encode_text(decoded)
                    except UnicodeDecodeError:
                        continue
                # 모든 인코딩 실패 시 에러 무시
                return text.decode('utf-8', errors='ignore')
            
            # 기타 타입인 경우
            else:
                return self._safe_encode_text(str(text))
                
        except Exception as e:
            self.logger.warning(f"텍스트 인코딩 처리 중 오류: {e}")
            # 최후의 수단: 안전한 문자만 유지
            try:
                safe_chars = ''.join(c for c in str(text) if ord(c) < 128 or c.isalnum() or c in ' .,!?-')
                return safe_chars
            except:
                return ""

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

    def _create_source_summary(self, doc: Dict[str, Any], index: int) -> str:
        """간결한 참고문헌 요약 생성"""
        text = doc['document']
        
        # 첫 번째 문장 또는 첫 100자 가져오기
        sentences = text.split('.')
        if len(sentences) > 0 and len(sentences[0]) > 20:
            summary = sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
        else:
            summary = text[:100] + "..." if len(text) > 100 else text
        
        # 요약 정리
        summary = summary.strip().replace('\n', ' ').replace('\r', ' ')
        summary = ' '.join(summary.split())  # 여분의 공백 제거
        
        return f"[참고 {index}] {summary}"

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
                return "I'm sorry, I cannot generate an answer based on the available information."
            
            return answer
            
        except Exception as e:
            self.logger.error(f"답변 생성 실패: {str(e)}")
            return f"An error occurred while generating the answer: {str(e)}"

    def _create_prompt(self, question: str, context: str) -> str:
        """프롬프트 생성"""
        prompt = f"""You are a medical AI that answers questions based solely on EAU (European Association of Urology) bladder cancer guidelines.

IMPORTANT INSTRUCTIONS:
- Answer ONLY based on the content explicitly stated in the guideline documents provided below
- If information is not found in the documents, respond with "The requested information is not available in the provided guidelines" or "I don't know"
- Do not answer based on external knowledge or speculation
- Do not provide general medical advice not found in the documents
- ALWAYS respond in English, regardless of the language of the question
- Be precise and cite specific sections when possible

EAU Guidelines Context:
{context}

Question: {question}

Answer based solely on the above guideline documents (respond in English):"""
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