#!/usr/bin/env python3
"""
유틸리티 모듈
DR-Bladder Agent의 공통 유틸리티 함수들을 제공합니다.
"""

import os
import sys
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import time
from datetime import datetime

# 색상 출력
from colorama import Fore, Style, init
init(autoreset=True)

def setup_logging(log_dir: str = None, log_level: str = "INFO", verbose: bool = False) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        log_dir: 로그 디렉토리 경로
        log_level: 로그 레벨
        verbose: 상세 로그 여부
    
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    if not log_dir:
        log_dir = str(Path.home() / ".dr-bladder" / "logs")
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 로그 레벨 설정
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 로거 생성
    logger = logging.getLogger("dr-bladder")
    logger.setLevel(level)
    
    # 핸들러가 이미 있으면 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러
    log_file = Path(log_dir) / f"dr-bladder-{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 콘솔 핸들러 (verbose 모드에서만)
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 파일 포맷터
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def check_system_requirements() -> bool:
    """
    시스템 요구사항 확인
    
    Returns:
        bool: 요구사항 만족 여부
    """
    try:
        print(f"{Fore.BLUE}🔍 시스템 요구사항 확인 중...{Style.RESET_ALL}")
        
        # Python 버전 확인
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print(f"{Fore.RED}❌ Python 3.8 이상이 필요합니다. 현재: {python_version.major}.{python_version.minor}{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}{Style.RESET_ALL}")
        
        # 필수 패키지 확인
        required_packages = [
            'torch', 'transformers', 'sentence_transformers', 
            'chromadb', 'PyPDF2', 'ollama', 'psutil', 'tqdm', 'colorama'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"{Fore.GREEN}✅ {package}{Style.RESET_ALL}")
            except ImportError:
                missing_packages.append(package)
                print(f"{Fore.RED}❌ {package} (미설치){Style.RESET_ALL}")
        
        if missing_packages:
            print(f"{Fore.RED}❌ 다음 패키지를 설치해주세요: {', '.join(missing_packages)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 설치 명령어: pip install {' '.join(missing_packages)}{Style.RESET_ALL}")
            return False
        
        # Ollama 설치 확인
        if not check_ollama_installation():
            print(f"{Fore.RED}❌ Ollama가 설치되지 않았습니다.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 설치 방법: https://ollama.ai{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}✅ 모든 시스템 요구사항을 만족합니다.{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}❌ 시스템 요구사항 확인 실패: {str(e)}{Style.RESET_ALL}")
        return False

def check_ollama_installation() -> bool:
    """
    Ollama 설치 확인
    
    Returns:
        bool: Ollama 설치 여부
    """
    try:
        # ollama 명령어 실행 가능 여부 확인
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print(f"{Fore.GREEN}✅ Ollama 설치됨: {result.stdout.strip()}{Style.RESET_ALL}")
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def check_ollama_service() -> bool:
    """
    Ollama 서비스 실행 상태 확인
    
    Returns:
        bool: 서비스 실행 여부
    """
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model_availability(model_name: str) -> bool:
    """
    Ollama 모델 사용 가능 여부 확인
    
    Args:
        model_name: 확인할 모델 이름
    
    Returns:
        bool: 모델 사용 가능 여부
    """
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        model_names = [model['name'] for model in models['models']]
        return model_name in model_names
    except:
        return False

def get_system_info() -> Dict[str, Any]:
    """
    시스템 정보 수집
    
    Returns:
        Dict: 시스템 정보
    """
    try:
        import psutil
        import torch
        
        # 기본 시스템 정보
        info = {
            "os": os.name,
            "platform": sys.platform,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "architecture": os.uname().machine if hasattr(os, 'uname') else 'unknown'
        }
        
        # CPU 정보
        info["cpu"] = {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent(interval=1),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        info["memory"] = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
        
        # GPU 정보
        info["gpu"] = {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info["gpu"]["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu"]["devices"].append({
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                })
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        info["disk"] = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        }
        
        return info
        
    except Exception as e:
        return {"error": str(e)}

def format_bytes(bytes_value: int) -> str:
    """
    바이트를 읽기 쉬운 형태로 변환
    
    Args:
        bytes_value: 바이트 값
    
    Returns:
        str: 포맷된 문자열
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

def format_time(seconds: float) -> str:
    """
    초를 읽기 쉬운 형태로 변환
    
    Args:
        seconds: 초 값
    
    Returns:
        str: 포맷된 문자열
    """
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}분 {remaining_seconds:.1f}초"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}시간 {int(remaining_minutes)}분"

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    진행률 바 생성
    
    Args:
        current: 현재 값
        total: 전체 값
        width: 바 너비
    
    Returns:
        str: 진행률 바
    """
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    empty = width - filled
    
    bar = "[" + "=" * filled + ">" + " " * empty + "]"
    percentage = progress * 100
    
    return f"{bar} {percentage:.1f}% ({current}/{total})"

def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    JSON 파일 저장
    
    Args:
        data: 저장할 데이터
        file_path: 파일 경로
        indent: 들여쓰기
    
    Returns:
        bool: 저장 성공 여부
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"JSON 저장 실패: {str(e)}")
        return False

def load_json(file_path: str) -> Optional[Any]:
    """
    JSON 파일 로드
    
    Args:
        file_path: 파일 경로
    
    Returns:
        Optional[Any]: 로드된 데이터 또는 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"JSON 로드 실패: {str(e)}")
        return None

def ensure_directory(directory: str) -> bool:
    """
    디렉토리 존재 확인 및 생성
    
    Args:
        directory: 디렉토리 경로
    
    Returns:
        bool: 성공 여부
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"디렉토리 생성 실패: {str(e)}")
        return False

def clean_text_for_display(text: str, max_length: int = 100) -> str:
    """
    텍스트를 화면 출력용으로 정리
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
    
    Returns:
        str: 정리된 텍스트
    """
    if not text:
        return ""
    
    # 줄바꿈 및 연속 공백 제거
    cleaned = ' '.join(text.split())
    
    # 길이 제한
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length-3] + "..."
    
    return cleaned

def validate_url(url: str) -> bool:
    """
    URL 유효성 검사
    
    Args:
        url: 확인할 URL
    
    Returns:
        bool: 유효성 여부
    """
    try:
        import re
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    except:
        return False

def test_ollama_connection(host: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Ollama 연결 테스트
    
    Args:
        host: Ollama 호스트
    
    Returns:
        Dict: 연결 테스트 결과
    """
    try:
        import requests
        
        # 연결 테스트
        response = requests.get(f"{host}/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            return {
                "connected": True,
                "models": models,
                "model_count": len(models),
                "message": "연결 성공"
            }
        else:
            return {
                "connected": False,
                "error": f"HTTP {response.status_code}",
                "message": "연결 실패"
            }
    
    except requests.exceptions.ConnectionError:
        return {
            "connected": False,
            "error": "연결 오류",
            "message": "Ollama 서버에 연결할 수 없습니다."
        }
    
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "message": "예상치 못한 오류가 발생했습니다."
        }

def benchmark_system() -> Dict[str, Any]:
    """
    시스템 성능 벤치마크
    
    Returns:
        Dict: 벤치마크 결과
    """
    try:
        import time
        import torch
        
        results = {}
        
        # CPU 벤치마크
        start_time = time.time()
        # 간단한 계산 작업
        result = sum(i * i for i in range(100000))
        cpu_time = time.time() - start_time
        results["cpu_benchmark"] = {
            "time": cpu_time,
            "ops_per_second": 100000 / cpu_time if cpu_time > 0 else 0
        }
        
        # GPU 벤치마크 (사용 가능한 경우)
        if torch.cuda.is_available():
            start_time = time.time()
            # 간단한 텐서 연산
            device = torch.device("cuda:0")
            tensor = torch.randn(1000, 1000, device=device)
            result = torch.mm(tensor, tensor)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            results["gpu_benchmark"] = {
                "time": gpu_time,
                "device": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0)
            }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def print_system_status():
    """시스템 상태 출력"""
    print(f"\n{Fore.CYAN}📊 시스템 상태{Style.RESET_ALL}")
    print("=" * 50)
    
    # 시스템 정보
    info = get_system_info()
    
    print(f"{Fore.YELLOW}🖥️  시스템 정보:{Style.RESET_ALL}")
    print(f"  • OS: {info.get('os', 'Unknown')} ({info.get('platform', 'Unknown')})")
    print(f"  • Python: {info.get('python_version', 'Unknown')}")
    print(f"  • 아키텍처: {info.get('architecture', 'Unknown')}")
    
    # CPU 정보
    cpu = info.get('cpu', {})
    print(f"\n{Fore.YELLOW}🔧 CPU 정보:{Style.RESET_ALL}")
    print(f"  • 코어 수: {cpu.get('count', 'Unknown')}")
    print(f"  • 사용률: {cpu.get('percent', 0):.1f}%")
    if cpu.get('frequency'):
        print(f"  • 주파수: {cpu.get('frequency', 0):.1f} MHz")
    
    # 메모리 정보
    memory = info.get('memory', {})
    print(f"\n{Fore.YELLOW}💾 메모리 정보:{Style.RESET_ALL}")
    print(f"  • 전체: {format_bytes(memory.get('total', 0))}")
    print(f"  • 사용 중: {format_bytes(memory.get('used', 0))} ({memory.get('percent', 0):.1f}%)")
    print(f"  • 사용 가능: {format_bytes(memory.get('available', 0))}")
    
    # GPU 정보
    gpu = info.get('gpu', {})
    print(f"\n{Fore.YELLOW}🎮 GPU 정보:{Style.RESET_ALL}")
    if gpu.get('available'):
        print(f"  • GPU 개수: {gpu.get('count', 0)}")
        for i, device in enumerate(gpu.get('devices', [])):
            print(f"  • GPU {i}: {device.get('name', 'Unknown')}")
            print(f"    - 전체 메모리: {format_bytes(device.get('total_memory', 0))}")
            print(f"    - 사용 중: {format_bytes(device.get('memory_allocated', 0))}")
    else:
        print(f"  • GPU 사용 불가")
    
    # 디스크 정보
    disk = info.get('disk', {})
    print(f"\n{Fore.YELLOW}💿 디스크 정보:{Style.RESET_ALL}")
    print(f"  • 전체: {format_bytes(disk.get('total', 0))}")
    print(f"  • 사용 중: {format_bytes(disk.get('used', 0))} ({disk.get('percent', 0):.1f}%)")
    print(f"  • 사용 가능: {format_bytes(disk.get('free', 0))}")
    
    # Ollama 연결 상태
    ollama_status = test_ollama_connection()
    print(f"\n{Fore.YELLOW}🤖 Ollama 상태:{Style.RESET_ALL}")
    if ollama_status.get('connected'):
        print(f"  • 상태: {Fore.GREEN}연결됨{Style.RESET_ALL}")
        print(f"  • 모델 수: {ollama_status.get('model_count', 0)}")
        models = ollama_status.get('models', [])
        if models:
            print(f"  • 설치된 모델: {', '.join(models[:3])}")
            if len(models) > 3:
                print(f"    ... 및 {len(models) - 3}개 더")
    else:
        print(f"  • 상태: {Fore.RED}연결 안됨{Style.RESET_ALL}")
        print(f"  • 오류: {ollama_status.get('error', 'Unknown')}")
    
    print()

if __name__ == "__main__":
    # 테스트 실행
    print_system_status()