#!/usr/bin/env python3
"""
설정 관리 모듈
DR-Bladder Agent의 모든 설정을 관리합니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """DR-Bladder Agent 설정 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        설정 초기화
        
        Args:
            config_path: 설정 파일 경로 (기본값: ~/.dr-bladder/config.json)
        """
        # 기본 설정
        self.home_dir = Path.home() / ".dr-bladder"
        self.config_path = config_path or str(self.home_dir / "config.json")
        self.cache_dir = str(self.home_dir / "cache")
        self.log_dir = str(self.home_dir / "logs")
        
        # 디렉토리 생성
        self.home_dir.mkdir(exist_ok=True)
        Path(self.cache_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        
        # 기본 설정값
        self._load_default_config()
        
        # 설정 파일 로드
        self.load_config()
    
    def _load_default_config(self):
        """기본 설정값 로드"""
        # Ollama 설정
        self.ollama_host = "http://localhost:11434"
        self.model_name = "qwen2.5:1.5b"
        
        # 임베딩 모델 설정
        self.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        
        # PDF 파일 경로
        self.pdf_path = str(Path(__file__).parent / "files" / "EAU-Guidelines-on-Non-muscle-invasive-Bladder-Cancer-2025.pdf")
        
        # RAG 설정
        self.top_k = 3  # 검색할 문서 수
        # 벡터화 배치 크기 (Windows에서는 더 작은 값 사용)
        self.batch_size = 16 if os.name == 'nt' else 32
        
        # 모델 생성 설정
        self.temperature = 0.7
        self.top_k_generate = 40
        self.top_p = 0.9
        self.max_tokens = 1000
        
        # 기타 설정
        self.verbose = False
        self.log_level = "INFO"
        self.language = "ko"
        
        # 시스템 설정
        self.gpu_memory_limit = 0.8  # GPU 메모리 사용 제한 (80%)
        self.cpu_cores = os.cpu_count()
        
    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 설정 적용
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                logging.info(f"설정 파일 로드 완료: {self.config_path}")
            else:
                # 기본 설정 파일 생성
                self.save_config()
                logging.info(f"기본 설정 파일 생성: {self.config_path}")
                
        except Exception as e:
            logging.error(f"설정 파일 로드 실패: {str(e)}")
            logging.info("기본 설정을 사용합니다.")
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            config_data = {
                # Ollama 설정
                "ollama_host": self.ollama_host,
                "model_name": self.model_name,
                
                # 임베딩 모델 설정
                "embedding_model": self.embedding_model,
                
                # PDF 파일 경로
                "pdf_path": self.pdf_path,
                
                # RAG 설정
                "top_k": self.top_k,
                "batch_size": self.batch_size,
                
                # 모델 생성 설정
                "temperature": self.temperature,
                "top_k_generate": self.top_k_generate,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                
                # 기타 설정
                "verbose": self.verbose,
                "log_level": self.log_level,
                "language": self.language,
                
                # 시스템 설정
                "gpu_memory_limit": self.gpu_memory_limit,
                "cpu_cores": self.cpu_cores
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"설정 파일 저장 완료: {self.config_path}")
            
        except Exception as e:
            logging.error(f"설정 파일 저장 실패: {str(e)}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return {
            "ollama_host": self.ollama_host,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "pdf_path": self.pdf_path,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_k_generate": self.top_k_generate,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "verbose": self.verbose,
            "log_level": self.log_level,
            "language": self.language,
            "gpu_memory_limit": self.gpu_memory_limit,
            "cpu_cores": self.cpu_cores,
            "cache_dir": self.cache_dir,
            "log_dir": self.log_dir
        }
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 설정 파일 저장
        self.save_config()
    
    def reset_config(self):
        """설정 초기화"""
        self._load_default_config()
        self.save_config()
        logging.info("설정을 기본값으로 초기화했습니다.")
    
    def validate_config(self) -> Dict[str, Any]:
        """설정 유효성 검사"""
        issues = []
        
        # PDF 파일 존재 확인
        if not os.path.exists(self.pdf_path):
            issues.append(f"PDF 파일이 존재하지 않습니다: {self.pdf_path}")
        
        # 디렉토리 존재 확인
        if not os.path.exists(self.cache_dir):
            issues.append(f"캐시 디렉토리가 존재하지 않습니다: {self.cache_dir}")
        
        if not os.path.exists(self.log_dir):
            issues.append(f"로그 디렉토리가 존재하지 않습니다: {self.log_dir}")
        
        # 숫자 범위 확인
        if not 0 < self.temperature <= 2.0:
            issues.append(f"temperature 값이 유효하지 않습니다: {self.temperature} (0 < temperature <= 2.0)")
        
        if not 0 < self.top_p <= 1.0:
            issues.append(f"top_p 값이 유효하지 않습니다: {self.top_p} (0 < top_p <= 1.0)")
        
        if not 1 <= self.top_k <= 20:
            issues.append(f"top_k 값이 유효하지 않습니다: {self.top_k} (1 <= top_k <= 20)")
        
        if not 1 <= self.batch_size <= 128:
            issues.append(f"batch_size 값이 유효하지 않습니다: {self.batch_size} (1 <= batch_size <= 128)")
        
        if not 100 <= self.max_tokens <= 4000:
            issues.append(f"max_tokens 값이 유효하지 않습니다: {self.max_tokens} (100 <= max_tokens <= 4000)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "ollama_model": self.model_name,
            "embedding_model": self.embedding_model,
            "ollama_host": self.ollama_host,
            "generation_params": {
                "temperature": self.temperature,
                "top_k": self.top_k_generate,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        import psutil
        import torch
        
        # GPU 정보
        gpu_info = {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            gpu_info["memory_reserved"] = torch.cuda.memory_reserved(0)
            gpu_info["memory_allocated"] = torch.cuda.memory_allocated(0)
        
        # CPU 정보
        cpu_info = {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent(interval=1),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
        
        return {
            "gpu": gpu_info,
            "cpu": cpu_info,
            "memory": memory_info,
            "platform": os.name,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"Config(model={self.model_name}, host={self.ollama_host}, pdf={os.path.basename(self.pdf_path)})"
    
    def __repr__(self) -> str:
        """객체 표현"""
        return self.__str__()