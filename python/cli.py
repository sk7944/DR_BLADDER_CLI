#!/usr/bin/env python3
"""
DR-Bladder-CLI
방광암 EAU 가이드라인 AI Agent - Ollama Qwen 기반 독립 CLI
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# 색상 출력을 위한 colorama
from colorama import init, Fore, Back, Style
init(autoreset=True)

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_agent import BladderCancerAgent
from config import Config
from utils import setup_logging, check_system_requirements

class BladderCLI:
    def __init__(self):
        self.agent = None
        self.config = Config()
        self.logger = setup_logging()
        
    def print_banner(self):
        """Print banner"""
        banner = f"""
{Fore.CYAN}╔════════════════════════════════════════════════════════════════╗
║                         DR-Bladder-CLI                         ║
║               Bladder Cancer EAU Guidelines AI Agent           ║
║                                                                ║
║                 Powered by Ollama + Qwen2.5 + RAG              ║
╚════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)

    def print_help(self):
        """Print help text"""
        help_text = f"""
{Fore.YELLOW}Usage:{Style.RESET_ALL}

  {Fore.GREEN}dr-bladder query "question"{Style.RESET_ALL}
    Ask bladder cancer related questions
    Example: dr-bladder query "What are BCG side effects?"

  {Fore.GREEN}dr-bladder chat{Style.RESET_ALL}
    Start interactive mode
    
  {Fore.GREEN}dr-bladder init{Style.RESET_ALL}
    Initialize system and install model
    
  {Fore.GREEN}dr-bladder status{Style.RESET_ALL}
    Check system status
    
  {Fore.GREEN}dr-bladder config{Style.RESET_ALL}
    Edit configuration file
    
  {Fore.GREEN}dr-bladder --help{Style.RESET_ALL}
    Show this help

{Fore.YELLOW}Example Questions:{Style.RESET_ALL}
  • "What are the side effects of BCG treatment?"
  • "What are the risk factors for bladder cancer recurrence?"
  • "How to manage after TURBT surgery?"
  • "What are the indications for BCG therapy?"

{Fore.YELLOW}Configuration:{Style.RESET_ALL}
  • Config file: ~/.dr-bladder/config.json
  • Log files: ~/.dr-bladder/logs/
  • Cache directory: ~/.dr-bladder/cache/
"""
        print(help_text)

    def init_agent(self) -> bool:
        """에이전트 초기화"""
        try:
            print(f"{Fore.BLUE}Initializing agent...{Style.RESET_ALL}")
            
            # 시스템 요구사항 검증
            if not check_system_requirements():
                print(f"{Fore.RED}System requirements not satisfied.{Style.RESET_ALL}")
                return False
            
            # 에이전트 생성
            self.agent = BladderCancerAgent(self.config)
            
            # 초기화
            if not self.agent.initialize():
                print(f"{Fore.RED}Agent initialization failed{Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}Agent initialization completed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Agent initialization error: {str(e)}{Style.RESET_ALL}")
            self.logger.error(f"Agent initialization failed: {str(e)}")
            return False

    def query(self, question: str) -> bool:
        """단일 질문 처리"""
        if not self.agent:
            if not self.init_agent():
                return False
        
        try:
            print(f"\n{Fore.YELLOW}Question: {question}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Generating answer...{Style.RESET_ALL}")
            
            # 질문 처리
            response = self.agent.ask_question(question)
            
            if response and response.get('success'):
                print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
                
                # 답변을 구조적으로 출력
                answer_text = response['answer']
                self._print_structured_answer(answer_text)
                
                # 간단한 참조 표시
                if response.get('sources'):
                    print(f"\n{Fore.CYAN}Referenced {len(response['sources'])} sections from EAU Guidelines{Style.RESET_ALL}")
                
                return True
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                print(f"{Fore.RED}Answer generation failed: {error_msg}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}Question processing error: {str(e)}{Style.RESET_ALL}")
            self.logger.error(f"Query processing failed: {str(e)}")
            return False

    def chat_mode(self):
        """대화형 모드"""
        if not self.agent:
            if not self.init_agent():
                return
        
        self.print_banner()
        print(f"{Fore.CYAN}Interactive mode started (Exit: 'quit', 'exit', 'q'){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Example: What are BCG side effects?{Style.RESET_ALL}\n")
        
        while True:
            try:
                # 사용자 입력 받기
                question = input(f"{Fore.GREEN}Question: {Style.RESET_ALL}").strip()
                
                # 종료 명령어 확인
                if question.lower() in ['quit', 'exit', 'q', '종료']:
                    print(f"{Fore.YELLOW}Ending conversation.{Style.RESET_ALL}")
                    break
                
                # 빈 입력 처리
                if not question:
                    continue
                
                # 질문 처리
                print(f"{Fore.BLUE}Generating answer...{Style.RESET_ALL}")
                response = self.agent.ask_question(question)
                
                if response and response.get('success'):
                    print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
                    
                    # 답변을 구조적으로 출력
                    answer_text = response['answer']
                    self._print_structured_answer(answer_text)
                    
                    # 간단한 참조 표시
                    if response.get('sources'):
                        print(f"\n{Fore.CYAN}Referenced {len(response['sources'])} sections from EAU Guidelines{Style.RESET_ALL}")
                else:
                    error_msg = response.get('error', 'Unknown error') if response else 'No response'
                    print(f"{Fore.RED}Answer generation failed: {error_msg}{Style.RESET_ALL}")
                
                print()  # 빈 줄 추가
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Ending conversation.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error occurred: {str(e)}{Style.RESET_ALL}")
                self.logger.error(f"Chat mode error: {str(e)}")

    def init_system(self):
        """시스템 초기화"""
        print(f"{Fore.BLUE}Starting system initialization...{Style.RESET_ALL}")
        
        # 시스템 요구사항 검증
        if not check_system_requirements():
            print(f"{Fore.RED}System requirements not satisfied.{Style.RESET_ALL}")
            return False
        
        # 에이전트 생성 및 초기화
        try:
            self.agent = BladderCancerAgent(self.config)
            if self.agent.initialize():
                print(f"{Fore.GREEN}System initialization completed{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}System initialization failed{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}System initialization error: {str(e)}{Style.RESET_ALL}")
            return False

    def show_status(self):
        """시스템 상태 확인"""
        print(f"{Fore.BLUE}Checking system status...{Style.RESET_ALL}")
        
        try:
            # 시스템 요구사항 검증
            requirements_ok = check_system_requirements()
            
            # 에이전트 상태 확인
            if not self.agent:
                self.agent = BladderCancerAgent(self.config)
            
            status = self.agent.get_status()
            
            print(f"\n{Fore.CYAN}System Status:{Style.RESET_ALL}")
            print(f"  • System requirements: {'OK' if requirements_ok else 'Not satisfied'}")
            print(f"  • Ollama server: {'Connected' if status.get('ollama_connected') else 'Not connected'}")
            print(f"  • Qwen model: {'Available' if status.get('model_available') else 'Not available'}")
            print(f"  • PDF document: {'Loaded' if status.get('pdf_loaded') else 'Not loaded'}")
            print(f"  • Vector DB: {'Ready' if status.get('vectordb_ready') else 'Not ready'}")
            
            # 설정 정보
            print(f"\n{Fore.CYAN}Configuration:{Style.RESET_ALL}")
            print(f"  • Model: {self.config.model_name}")
            print(f"  • Config file: {self.config.config_path}")
            print(f"  • Cache directory: {self.config.cache_dir}")
            
        except Exception as e:
            print(f"{Fore.RED}Error checking status: {str(e)}{Style.RESET_ALL}")

    def _print_structured_answer(self, answer_text: str):
        """답변을 plain text로 그대로 출력"""
        # AI의 답변을 아무런 가공 없이 그대로 출력
        print(answer_text)

    def edit_config(self):
        """설정 파일 편집"""
        config_path = self.config.config_path
        
        print(f"{Fore.BLUE}Configuration file path: {config_path}{Style.RESET_ALL}")
        
        # 설정 파일이 없으면 생성
        if not os.path.exists(config_path):
            self.config.save_config()
            print(f"{Fore.GREEN}Default configuration file created.{Style.RESET_ALL}")
        
        # 시스템 기본 에디터로 열기
        try:
            os.system(f"{'notepad' if os.name == 'nt' else 'nano'} {config_path}")
        except Exception as e:
            print(f"{Fore.RED}Configuration file edit failed: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please edit manually: {config_path}{Style.RESET_ALL}")

    def run(self):
        """메인 실행 함수"""
        parser = argparse.ArgumentParser(
            description='DR-Bladder-CLI - 방광암 EAU 가이드라인 AI Agent',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 명령어 인자 추가
        parser.add_argument('command', nargs='?', choices=['query', 'chat', 'init', 'status', 'config'], 
                          help='실행할 명령어')
        parser.add_argument('question', nargs='?', help='Question content (when using query command)')
        parser.add_argument('--model', '-m', default='qwen2.5:1.5b', help='Model to use (default: qwen2.5:1.5b)')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose log output')
        parser.add_argument('--config', '-c', help='Configuration file path')
        
        args = parser.parse_args()
        
        # 설정 업데이트
        if args.config:
            self.config.config_path = args.config
        if args.model:
            self.config.model_name = args.model
        if args.verbose:
            self.config.verbose = True
        
        # 명령어 실행
        if args.command == 'query':
            if not args.question:
                print(f"{Fore.RED}Please enter a question.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Example: dr-bladder query \"What are BCG side effects?\"{Style.RESET_ALL}")
                return
            return self.query(args.question)
        
        elif args.command == 'chat':
            self.chat_mode()
        
        elif args.command == 'init':
            return self.init_system()
        
        elif args.command == 'status':
            self.show_status()
        
        elif args.command == 'config':
            self.edit_config()
        
        else:
            # 명령어가 없으면 도움말 출력
            self.print_help()

if __name__ == '__main__':
    cli = BladderCLI()
    cli.run()