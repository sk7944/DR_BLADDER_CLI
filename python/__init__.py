#!/usr/bin/env python3
"""
DR-Bladder-CLI
방광암 EAU 가이드라인 AI Agent - Ollama Qwen 기반 독립 CLI

이 패키지는 방광암 치료 가이드라인을 기반으로 한 AI Agent입니다.
"""

__version__ = "1.0.0"
__author__ = "DR-Bladder Team"
__email__ = "support@dr-bladder.com"
__description__ = "방광암 EAU 가이드라인 AI Agent - Ollama Qwen 기반 독립 CLI"

from .bladder_agent import BladderCancerAgent
from .config import Config
from .utils import setup_logging, check_system_requirements

__all__ = [
    "BladderCancerAgent",
    "Config", 
    "setup_logging",
    "check_system_requirements"
]