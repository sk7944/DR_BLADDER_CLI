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
║                    🏥 DR-Bladder-CLI                          ║
║              Bladder Cancer EAU Guidelines AI Agent           ║
║                                                                ║
║           Powered by Ollama + Qwen2.5 + RAG                   ║
╚════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)

    def print_help(self):
        """도움말 출력"""
        help_text = f"""
{Fore.YELLOW}🚀 사용법:{Style.RESET_ALL}

  {Fore.GREEN}dr-bladder query "질문내용"{Style.RESET_ALL}
    방광암 관련 질문하기
    예: dr-bladder query "BCG 치료의 부작용은?"

  {Fore.GREEN}dr-bladder chat{Style.RESET_ALL}
    대화형 모드로 시작
    
  {Fore.GREEN}dr-bladder init{Style.RESET_ALL}
    시스템 초기화 및 모델 설치
    
  {Fore.GREEN}dr-bladder status{Style.RESET_ALL}
    시스템 상태 확인
    
  {Fore.GREEN}dr-bladder config{Style.RESET_ALL}
    설정 파일 편집
    
  {Fore.GREEN}dr-bladder --help{Style.RESET_ALL}
    이 도움말 표시

{Fore.YELLOW}💡 예시 질문:{Style.RESET_ALL}
  • "BCG 치료의 부작용은 무엇인가요?"
  • "방광암의 재발 위험 요인에 대해 알려주세요."
  • "TURBT 수술 후 관리 방법은?"
  • "What are the indications for BCG therapy?"

{Fore.YELLOW}⚙️ 설정:{Style.RESET_ALL}
  • 설정 파일: ~/.dr-bladder/config.json
  • 로그 파일: ~/.dr-bladder/logs/
  • 캐시 디렉토리: ~/.dr-bladder/cache/
"""
        print(help_text)

    def init_agent(self) -> bool:
        """에이전트 초기화"""
        try:
            print(f"{Fore.BLUE}🔄 에이전트 초기화 중...{Style.RESET_ALL}")
            
            # 시스템 요구사항 검증
            if not check_system_requirements():
                print(f"{Fore.RED}❌ 시스템 요구사항을 만족하지 않습니다.{Style.RESET_ALL}")
                return False
            
            # 에이전트 생성
            self.agent = BladderCancerAgent(self.config)
            
            # 초기화
            if not self.agent.initialize():
                print(f"{Fore.RED}❌ 에이전트 초기화 실패{Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}✅ 에이전트 초기화 완료{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 에이전트 초기화 중 오류: {str(e)}{Style.RESET_ALL}")
            self.logger.error(f"Agent initialization failed: {str(e)}")
            return False

    def query(self, question: str) -> bool:
        """단일 질문 처리"""
        if not self.agent:
            if not self.init_agent():
                return False
        
        try:
            print(f"\n{Fore.YELLOW}🤔 질문: {question}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}🔍 답변 생성 중...{Style.RESET_ALL}")
            
            # 질문 처리
            response = self.agent.ask_question(question)
            
            if response and response.get('success'):
                print(f"\n{Fore.GREEN}🏥 답변:{Style.RESET_ALL}")
                print(f"{response['answer']}")
                
                # 간단한 참조 표시
                if response.get('sources'):
                    print(f"\n{Fore.CYAN}📚 EAU 가이드라인 {len(response['sources'])}개 섹션 참조{Style.RESET_ALL}")
                
                return True
            else:
                error_msg = response.get('error', '알 수 없는 오류') if response else '응답 없음'
                print(f"{Fore.RED}❌ 답변 생성 실패: {error_msg}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ 질문 처리 중 오류: {str(e)}{Style.RESET_ALL}")
            self.logger.error(f"Query processing failed: {str(e)}")
            return False

    def chat_mode(self):
        """대화형 모드"""
        if not self.agent:
            if not self.init_agent():
                return
        
        self.print_banner()
        print(f"{Fore.CYAN}💬 대화형 모드 시작 (종료: 'quit', 'exit', 'q'){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 예시: BCG 치료의 부작용은?{Style.RESET_ALL}\n")
        
        while True:
            try:
                # 사용자 입력 받기
                question = input(f"{Fore.GREEN}🤔 질문: {Style.RESET_ALL}").strip()
                
                # 종료 명령어 확인
                if question.lower() in ['quit', 'exit', 'q', '종료']:
                    print(f"{Fore.YELLOW}👋 대화를 종료합니다.{Style.RESET_ALL}")
                    break
                
                # 빈 입력 처리
                if not question:
                    continue
                
                # 질문 처리
                print(f"{Fore.BLUE}🔍 답변 생성 중...{Style.RESET_ALL}")
                response = self.agent.ask_question(question)
                
                if response and response.get('success'):
                    print(f"\n{Fore.GREEN}🏥 답변:{Style.RESET_ALL}")
                    print(f"{response['answer']}")
                    
                    # 간단한 참조 표시
                    if response.get('sources'):
                        print(f"\n{Fore.CYAN}📚 EAU 가이드라인 {len(response['sources'])}개 섹션 참조{Style.RESET_ALL}")
                else:
                    error_msg = response.get('error', '알 수 없는 오류') if response else '응답 없음'
                    print(f"{Fore.RED}❌ 답변 생성 실패: {error_msg}{Style.RESET_ALL}")
                
                print()  # 빈 줄 추가
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}👋 대화를 종료합니다.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ 오류 발생: {str(e)}{Style.RESET_ALL}")
                self.logger.error(f"Chat mode error: {str(e)}")

    def init_system(self):
        """시스템 초기화"""
        print(f"{Fore.BLUE}🔄 시스템 초기화 시작...{Style.RESET_ALL}")
        
        # 시스템 요구사항 검증
        if not check_system_requirements():
            print(f"{Fore.RED}❌ 시스템 요구사항을 만족하지 않습니다.{Style.RESET_ALL}")
            return False
        
        # 에이전트 생성 및 초기화
        try:
            self.agent = BladderCancerAgent(self.config)
            if self.agent.initialize():
                print(f"{Fore.GREEN}✅ 시스템 초기화 완료{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ 시스템 초기화 실패{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}❌ 시스템 초기화 중 오류: {str(e)}{Style.RESET_ALL}")
            return False

    def show_status(self):
        """시스템 상태 확인"""
        print(f"{Fore.BLUE}🔍 시스템 상태 확인 중...{Style.RESET_ALL}")
        
        try:
            # 시스템 요구사항 검증
            requirements_ok = check_system_requirements()
            
            # 에이전트 상태 확인
            if not self.agent:
                self.agent = BladderCancerAgent(self.config)
            
            status = self.agent.get_status()
            
            print(f"\n{Fore.CYAN}📊 시스템 상태:{Style.RESET_ALL}")
            print(f"  • 시스템 요구사항: {'✅ OK' if requirements_ok else '❌ 미충족'}")
            print(f"  • Ollama 서버: {'✅ 연결됨' if status.get('ollama_connected') else '❌ 연결 안됨'}")
            print(f"  • Qwen 모델: {'✅ 사용 가능' if status.get('model_available') else '❌ 사용 불가'}")
            print(f"  • PDF 문서: {'✅ 로드됨' if status.get('pdf_loaded') else '❌ 로드 안됨'}")
            print(f"  • 벡터 DB: {'✅ 준비됨' if status.get('vectordb_ready') else '❌ 준비 안됨'}")
            
            # 설정 정보
            print(f"\n{Fore.CYAN}⚙️ 설정 정보:{Style.RESET_ALL}")
            print(f"  • 모델: {self.config.model_name}")
            print(f"  • 설정 파일: {self.config.config_path}")
            print(f"  • 캐시 디렉토리: {self.config.cache_dir}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 상태 확인 중 오류: {str(e)}{Style.RESET_ALL}")

    def edit_config(self):
        """설정 파일 편집"""
        config_path = self.config.config_path
        
        print(f"{Fore.BLUE}⚙️ 설정 파일 경로: {config_path}{Style.RESET_ALL}")
        
        # 설정 파일이 없으면 생성
        if not os.path.exists(config_path):
            self.config.save_config()
            print(f"{Fore.GREEN}✅ 기본 설정 파일을 생성했습니다.{Style.RESET_ALL}")
        
        # 시스템 기본 에디터로 열기
        try:
            os.system(f"{'notepad' if os.name == 'nt' else 'nano'} {config_path}")
        except Exception as e:
            print(f"{Fore.RED}❌ 설정 파일 편집 실패: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 직접 편집하세요: {config_path}{Style.RESET_ALL}")

    def run(self):
        """메인 실행 함수"""
        parser = argparse.ArgumentParser(
            description='DR-Bladder-CLI - 방광암 EAU 가이드라인 AI Agent',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 명령어 인자 추가
        parser.add_argument('command', nargs='?', choices=['query', 'chat', 'init', 'status', 'config'], 
                          help='실행할 명령어')
        parser.add_argument('question', nargs='?', help='질문 내용 (query 명령어 사용시)')
        parser.add_argument('--model', '-m', default='qwen2.5:0.5b', help='사용할 모델 (기본: qwen2.5:0.5b)')
        parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
        parser.add_argument('--config', '-c', help='설정 파일 경로')
        
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
                print(f"{Fore.RED}❌ 질문을 입력해주세요.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}💡 예시: dr-bladder query \"BCG 치료의 부작용은?\"{Style.RESET_ALL}")
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