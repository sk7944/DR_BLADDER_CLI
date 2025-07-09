#!/usr/bin/env python3
"""
BladderCancerAgent - 방광암 EAU 가이드라인 AI Agent 핵심 클래스
Ollama Qwen 모델과 RAG 기능을 통합한 독립 에이전트
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import unicodedata
import re
import time
import json
import pickle
import numpy as np

# 필수 라이브러리
import ollama
import torch
import PyPDF2
from sentence_transformers import SentenceTransformer
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
        self.vector_store = None  # ChromaDB 대신 간단한 벡터 저장소
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
        """UTF-8 인코딩 통일 설정"""
        try:
            import locale
            import sys
            
            # 환경 변수를 UTF-8로 통일
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['LANG'] = 'en_US.UTF-8'
            os.environ['LC_ALL'] = 'en_US.UTF-8'
            
            # Windows에서 콘솔 인코딩 설정
            if os.name == 'nt':
                try:
                    import codecs
                    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
                    # Windows 콘솔 UTF-8 모드 활성화
                    os.system('chcp 65001 > nul')
                except Exception as e:
                    self.logger.warning(f"Windows 콘솔 인코딩 설정 실패: {e}")
            
            # Python 스트림 인코딩 설정
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
            if hasattr(sys.stdin, 'reconfigure'):
                sys.stdin.reconfigure(encoding='utf-8', errors='ignore')
            
            # 로케일 설정 (UTF-8 우선)
            locale_candidates = [
                'en_US.UTF-8',
                'C.UTF-8',
                'ko_KR.UTF-8',
                'Korean_Korea.utf8' if os.name == 'nt' else None
            ]
            
            for loc in locale_candidates:
                if loc:
                    try:
                        locale.setlocale(locale.LC_ALL, loc)
                        self.logger.info(f"로케일 설정 성공: {loc}")
                        break
                    except:
                        continue
                        
        except Exception as e:
            self.logger.warning(f"인코딩 설정 중 오류: {e}")
        
        # 기본 인코딩 확인
        try:
            import sys
            self.logger.info(f"시스템 기본 인코딩: {sys.getdefaultencoding()}")
            self.logger.info(f"파일 시스템 인코딩: {sys.getfilesystemencoding()}")
        except:
            pass

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
            
            # 3. 벡터 저장소 초기화
            if not self._init_vector_store():
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
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
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

    def _init_vector_store(self) -> bool:
        """간단한 벡터 저장소 초기화"""
        try:
            self.logger.info("벡터 저장소 초기화 중...")
            print("간단한 벡터 저장소 사용 (ChromaDB 대신)")
            
            # 간단한 벡터 저장소 구조
            self.vector_store = {
                'embeddings': [],
                'documents': [],
                'ids': []
            }
            
            self.logger.info("벡터 저장소 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"벡터 저장소 초기화 실패: {str(e)}")
            return False

    def _load_pdf_and_vectorize(self) -> bool:
        """PDF 로드 및 벡터화"""
        try:
            self.logger.info("PDF 로드 및 벡터화 시작...")
            print("PDF 문서 처리를 시작합니다...")
            
            # PDF 파일 경로
            pdf_path = Path(self.config.pdf_path)
            if not pdf_path.exists():
                self.logger.error(f"PDF 파일이 없습니다: {pdf_path}")
                print(f"ERROR: PDF 파일을 찾을 수 없습니다: {pdf_path}")
                return False
            
            print(f"PDF 파일 크기: {pdf_path.stat().st_size / (1024*1024):.1f}MB")
            
            # PDF 텍스트 추출
            print("PDF에서 텍스트 추출 중...")
            documents = self._extract_pdf_text(pdf_path)
            if not documents:
                self.logger.error("PDF에서 텍스트 추출 실패")
                print("ERROR: PDF에서 텍스트 추출에 실패했습니다.")
                return False
            
            print(f"PDF 텍스트 추출 완료: {len(documents)}개 페이지")
            
            # 방광암 관련 문서만 필터링
            print("방광암 관련 문서 필터링 중...")
            filtered_docs = self._filter_relevant_documents(documents)
            if not filtered_docs:
                self.logger.error("방광암 관련 문서가 없습니다")
                print("ERROR: 방광암 관련 문서를 찾을 수 없습니다.")
                return False
            
            print(f"관련 문서 필터링 완료: {len(filtered_docs)}개 문서")
            
            # 벡터화 및 데이터베이스 저장
            print("문서 벡터화 및 데이터베이스 저장 중...")
            self._vectorize_and_store(filtered_docs)
            
            self.pdf_loaded = True
            self.logger.info(f"PDF 벡터화 완료: {len(filtered_docs)}개 문서")
            print("PDF 벡터화가 완료되었습니다!")
            return True
            
        except Exception as e:
            self.logger.error(f"PDF 로드 및 벡터화 실패: {str(e)}")
            print(f"ERROR: PDF 처리 중 오류가 발생했습니다: {str(e)}")
            return False

    def _extract_pdf_text(self, pdf_path: Path) -> List[str]:
        """PDF에서 텍스트 추출"""
        try:
            documents = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                total_pages = len(pdf_reader.pages)
                print(f"총 {total_pages}개 페이지 처리 시작...")
                
                for page_num in tqdm(range(total_pages), desc="페이지 처리"):
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
                
                print(f"페이지 처리 완료: {len(documents)}개 문서 추출")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 실패: {str(e)}")
            print(f"ERROR: PDF 텍스트 추출 중 오류: {str(e)}")
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
            # 배치 크기 설정 (더 작은 배치로 시작)
            batch_size = 8  # 기본값을 더 작게 설정
            if os.name == 'nt':  # Windows
                batch_size = min(batch_size, 4)  # Windows에서는 4로 제한
            
            print(f"문서 벡터화 시작... (총 {len(documents)}개 문서, 배치 크기: {batch_size})")
            
            for i in tqdm(range(0, len(documents), batch_size), desc="문서 벡터화"):
                try:
                    batch = documents[i:i+batch_size]
                    batch_num = i // batch_size + 1
                    
                    print(f"배치 {batch_num} 처리 중... ({len(batch)}개 문서)")
                    
                    # 메모리 사용량 확인
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 70:
                        print(f"WARNING: 메모리 사용량이 높습니다 ({memory_percent:.1f}%). 배치 크기를 줄입니다.")
                        batch_size = max(1, batch_size // 2)
                        batch = documents[i:i+batch_size]
                    
                    # GPU 메모리 정리 (사용 가능한 경우)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 임베딩 생성 (타임아웃 추가)
                    print(f"배치 {batch_num} 임베딩 생성 중...")
                    start_time = time.time()
                    
                    # 문서 길이 제한 (너무 긴 문서는 잘라내기)
                    max_length = 1000
                    truncated_batch = []
                    for doc in batch:
                        if len(doc) > max_length:
                            truncated_batch.append(doc[:max_length] + "...")
                        else:
                            truncated_batch.append(doc)
                    
                    embeddings = self.embedding_model.encode(
                        truncated_batch, 
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        batch_size=min(4, len(truncated_batch))  # 내부 배치 크기도 제한
                    )
                    
                    elapsed_time = time.time() - start_time
                    print(f"배치 {batch_num} 임베딩 완료 ({elapsed_time:.2f}초)")
                    
                    # 간단한 벡터 저장소에 저장
                    print(f"배치 {batch_num} 벡터 저장소에 저장 중...")
                    ids = [f"doc_{i+j}" for j in range(len(batch))]
                    
                    # 임베딩을 numpy array로 변환 후 리스트로 변환
                    import numpy as np
                    if hasattr(embeddings, 'numpy'):
                        embeddings_list = embeddings.numpy().tolist()
                    elif isinstance(embeddings, np.ndarray):
                        embeddings_list = embeddings.tolist()
                    else:
                        embeddings_list = embeddings.tolist()
                    
                    # 간단한 벡터 저장소에 저장
                    success_count = 0
                    for j, (embedding, doc, doc_id) in enumerate(zip(embeddings_list, batch, ids)):
                        try:
                            print(f"  문서 {j+1}/{len(batch)} 저장 중...")
                            
                            # 문서 내용 정리
                            cleaned_doc = self._safe_encode_text(doc)
                            if len(cleaned_doc) > 4000:
                                cleaned_doc = cleaned_doc[:4000] + "..."
                            
                            # 임베딩 검증
                            if not isinstance(embedding, list) or len(embedding) == 0:
                                print(f"  문서 {j+1} 임베딩 오류, 건너뜀")
                                continue
                            
                            # 간단한 벡터 저장소에 저장
                            self.vector_store['embeddings'].append(embedding)
                            self.vector_store['documents'].append(cleaned_doc)
                            self.vector_store['ids'].append(doc_id)
                            
                            print(f"  문서 {j+1} 저장 완료")
                            success_count += 1
                            
                        except Exception as doc_error:
                            print(f"  문서 {j+1} 저장 실패: {str(doc_error)}")
                            self.logger.error(f"문서 {doc_id} 저장 실패: {str(doc_error)}")
                            continue
                    
                    print(f"  벡터 저장소 저장 완료: {success_count}/{len(batch)}개 문서 저장됨")
                    
                    print(f"배치 {batch_num} 완료")
                    
                    # 배치 간 잠시 대기 (메모리 안정화)
                    time.sleep(0.1)
                    
                except MemoryError as e:
                    self.logger.error(f"메모리 부족 오류 (배치 {batch_num}): {str(e)}")
                    print(f"ERROR: 메모리 부족으로 인해 벡터화에 실패했습니다. 배치 크기를 줄여서 다시 시도해주세요.")
                    # 배치 크기를 더 줄여서 재시도
                    batch_size = max(1, batch_size // 2)
                    if batch_size == 1:
                        raise
                    continue
                except Exception as e:
                    self.logger.error(f"배치 {batch_num} 처리 중 오류: {str(e)}")
                    print(f"ERROR: 배치 {batch_num} 처리 중 오류가 발생했습니다: {str(e)}")
                    # 개별 배치 오류는 건너뛰고 계속 진행
                    continue
                    
        except Exception as e:
            self.logger.error(f"벡터화 및 저장 실패: {str(e)}")
            print(f"ERROR: 문서 벡터화 중 오류가 발생했습니다: {str(e)}")
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
        """관련 문서 검색 (간단한 벡터 저장소 사용)"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])
            query_vector = query_embedding[0] if hasattr(query_embedding, 'shape') else query_embedding
            
            # 코사인 유사도 계산
            similarities = []
            for i, doc_embedding in enumerate(self.vector_store['embeddings']):
                # 코사인 유사도 계산
                dot_product = np.dot(query_vector, doc_embedding)
                norm_query = np.linalg.norm(query_vector)
                norm_doc = np.linalg.norm(doc_embedding)
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                else:
                    similarity = 0
                
                similarities.append({
                    "index": i,
                    "similarity": similarity,
                    "document": self.vector_store['documents'][i],
                    "id": self.vector_store['ids'][i]
                })
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 상위 결과 선택
            top_results = similarities[:self.config.top_k]
            
            # 결과 정리
            relevant_docs = []
            for result in top_results:
                relevant_docs.append({
                    "document": result['document'],
                    "distance": 1 - result['similarity'],  # 거리는 1 - 유사도
                    "id": result['id']
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
            
            # 답변을 그대로 반환 (포맷팅 없이)
            return answer
            
        except Exception as e:
            self.logger.error(f"답변 생성 실패: {str(e)}")
            return f"An error occurred while generating the answer: {str(e)}"
    

    def _create_prompt(self, question: str, context: str) -> str:
        """프롬프트 생성"""
        # 질문 언어 감지
        is_korean = self._is_korean_question(question)
        
        if is_korean:
            prompt = f"""당신은 EAU (European Association of Urology) 방광암 가이드라인에 기반하여 질문에 답변하는 의료 AI입니다.

중요한 지시사항:
- 아래 제공된 가이드라인 문서에 명시적으로 기술된 내용만을 바탕으로 답변하세요
- 문서에서 정보를 찾을 수 없으면 "제공된 가이드라인에서 해당 정보를 찾을 수 없습니다" 또는 "모르겠습니다"라고 답변하세요
- 외부 지식이나 추측에 기반하여 답변하지 마세요
- 문서에 없는 일반적인 의료 조언을 제공하지 마세요
- 한국어로 질문하면 한국어로 답변하세요
- 정확하고 구체적으로 답변하며, 가능하면 특정 섹션을 인용하세요

중요한 형식 지시사항:
- 번호 목록을 사용할 때는 반드시 다음 형식을 사용하세요:
  올바른 형식: "다음과 같습니다:\n1. 첫 번째 항목\n2. 두 번째 항목"
  잘못된 형식: "다음과 같습니다:1. 첫 번째 항목2. 두 번째 항목"
- 각 번호는 새로운 줄에서 시작해야 합니다
- 번호 뒤에는 공백을 하나 넣으세요

EAU 가이드라인 컨텍스트:
{context}

질문: {question}

위의 가이드라인 문서만을 바탕으로 한국어로 답변하세요. 번호 목록 사용 시 올바른 형식을 따르세요:"""
        else:
            prompt = f"""You are a medical AI that answers questions based solely on EAU (European Association of Urology) bladder cancer guidelines.

IMPORTANT INSTRUCTIONS:
- Answer ONLY based on the content explicitly stated in the guideline documents provided below
- If information is not found in the documents, respond with "The requested information is not available in the provided guidelines" or "I don't know"
- Do not answer based on external knowledge or speculation
- Do not provide general medical advice not found in the documents
- Respond in English for English questions
- Be precise and cite specific sections when possible

IMPORTANT FORMATTING INSTRUCTIONS:
- When using numbered lists, use this exact format:
  Correct format: "factors include:\n1. First item\n2. Second item"
  Incorrect format: "factors include:1. First item2. Second item"
- Each number must start on a new line
- Put one space after each number

EAU Guidelines Context:
{context}

Question: {question}

Answer based solely on the above guideline documents (respond in English). Use correct formatting for numbered lists:"""
        
        return prompt
        
    def _is_korean_question(self, question: str) -> bool:
        """질문이 한국어인지 판단"""
        korean_chars = 0
        total_chars = 0
        
        for char in question:
            if char.isalpha():
                total_chars += 1
                if '\uac00' <= char <= '\ud7a3':  # 한글 유니코드 범위
                    korean_chars += 1
        
        if total_chars == 0:
            return False
        
        return korean_chars / total_chars > 0.3  # 30% 이상이 한글이면 한국어 질문

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
            
            # 벡터 저장소 상태 확인
            if self.vector_store:
                try:
                    count = len(self.vector_store['documents'])
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
            
            if hasattr(self, 'vector_store'):
                del self.vector_store
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {str(e)}")