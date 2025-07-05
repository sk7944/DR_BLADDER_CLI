#!/usr/bin/env python3
"""
DR-Bladder-CLI
Bladder Cancer EAU Guidelines AI Agent - Ollama Qwen based independent CLI
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Color output with colorama
from colorama import init, Fore, Back, Style
init(autoreset=True)

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_agent_en import BladderCancerAgent
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
        """Print help"""
        help_text = f"""
{Fore.YELLOW}🚀 Usage:{Style.RESET_ALL}

  {Fore.GREEN}dr-bladder query "your question"{Style.RESET_ALL}
    Ask bladder cancer related questions
    Example: dr-bladder query "What are BCG side effects?"

  {Fore.GREEN}dr-bladder chat{Style.RESET_ALL}
    Start interactive chat mode
    
  {Fore.GREEN}dr-bladder init{Style.RESET_ALL}
    Initialize system and install models
    
  {Fore.GREEN}dr-bladder status{Style.RESET_ALL}
    Check system status
    
  {Fore.GREEN}dr-bladder config{Style.RESET_ALL}
    Edit configuration file
    
  {Fore.GREEN}dr-bladder --help{Style.RESET_ALL}
    Show this help message

{Fore.YELLOW}💡 Example questions:{Style.RESET_ALL}
  • "What are the side effects of BCG therapy?"
  • "Tell me about bladder cancer recurrence risk factors"
  • "What is the post-TURBT management?"
  • "What are the indications for BCG therapy?"

{Fore.YELLOW}⚙️ Configuration:{Style.RESET_ALL}
  • Config file: ~/.dr-bladder/config.json
  • Log files: ~/.dr-bladder/logs/
  • Cache directory: ~/.dr-bladder/cache/
"""
        print(help_text)

    def init_agent(self) -> bool:
        """Initialize agent"""
        try:
            print(f"{Fore.BLUE}🔄 Initializing agent...{Style.RESET_ALL}")
            
            # Check system requirements
            if not check_system_requirements():
                print(f"{Fore.RED}❌ System requirements not met.{Style.RESET_ALL}")
                return False
            
            # Create agent
            self.agent = BladderCancerAgent(self.config)
            
            # Initialize
            if not self.agent.initialize():
                print(f"{Fore.RED}❌ Agent initialization failed{Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}✅ Agent initialization completed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error during agent initialization: {str(e)}{Style.RESET_ALL}")
            self.logger.error(f"Agent initialization failed: {str(e)}")
            return False

    def query(self, question: str) -> bool:
        """Process single question"""
        if not self.agent:
            if not self.init_agent():
                return False
        
        try:
            print(f"\\n{Fore.YELLOW}🤔 Question: {question}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}🔍 Generating answer...{Style.RESET_ALL}")
            
            # Process question
            response = self.agent.ask_question(question)
            
            if response and response.get('success'):
                print(f"\\n{Fore.GREEN}🏥 Answer:{Style.RESET_ALL}")
                print(f"{response['answer']}")
                
                # Show reference documents
                if response.get('sources'):
                    print(f"\\n{Fore.CYAN}📚 Reference documents:{Style.RESET_ALL}")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source}")
                
                return True
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                print(f"{Fore.RED}❌ Answer generation failed: {error_msg}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error during question processing: {str(e)}{Style.RESET_ALL}")
            self.logger.error(f"Query processing failed: {str(e)}")
            return False

    def chat_mode(self):
        """Interactive chat mode"""
        if not self.agent:
            if not self.init_agent():
                return
        
        self.print_banner()
        print(f"{Fore.CYAN}💬 Starting interactive mode (quit: 'quit', 'exit', 'q'){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Example: What are BCG side effects?{Style.RESET_ALL}\\n")
        
        while True:
            try:
                # Get user input
                question = input(f"{Fore.GREEN}🤔 Question: {Style.RESET_ALL}").strip()
                
                # Check exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print(f"{Fore.YELLOW}👋 Ending conversation.{Style.RESET_ALL}")
                    break
                
                # Handle empty input
                if not question:
                    continue
                
                # Process question
                print(f"{Fore.BLUE}🔍 Generating answer...{Style.RESET_ALL}")
                response = self.agent.ask_question(question)
                
                if response and response.get('success'):
                    print(f"\\n{Fore.GREEN}🏥 Answer:{Style.RESET_ALL}")
                    print(f"{response['answer']}")
                    
                    # Show reference documents
                    if response.get('sources'):
                        print(f"\\n{Fore.CYAN}📚 Reference documents:{Style.RESET_ALL}")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"  {i}. {source}")
                else:
                    error_msg = response.get('error', 'Unknown error') if response else 'No response'
                    print(f"{Fore.RED}❌ Answer generation failed: {error_msg}{Style.RESET_ALL}")
                
                print()  # Add empty line
                
            except KeyboardInterrupt:
                print(f"\\n{Fore.YELLOW}👋 Ending conversation.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Error occurred: {str(e)}{Style.RESET_ALL}")
                self.logger.error(f"Chat mode error: {str(e)}")

    def init_system(self):
        """Initialize system"""
        print(f"{Fore.BLUE}🔄 Starting system initialization...{Style.RESET_ALL}")
        
        # Check system requirements
        if not check_system_requirements():
            print(f"{Fore.RED}❌ System requirements not met.{Style.RESET_ALL}")
            return False
        
        # Create and initialize agent
        try:
            self.agent = BladderCancerAgent(self.config)
            if self.agent.initialize():
                print(f"{Fore.GREEN}✅ System initialization completed{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ System initialization failed{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"{Fore.RED}❌ Error during system initialization: {str(e)}{Style.RESET_ALL}")
            return False

    def show_status(self):
        """Check system status"""
        print(f"{Fore.BLUE}🔍 Checking system status...{Style.RESET_ALL}")
        
        try:
            # Check system requirements
            requirements_ok = check_system_requirements()
            
            # Check agent status
            if not self.agent:
                self.agent = BladderCancerAgent(self.config)
            
            status = self.agent.get_status()
            
            print(f"\\n{Fore.CYAN}📊 System Status:{Style.RESET_ALL}")
            print(f"  • System requirements: {'✅ OK' if requirements_ok else '❌ Not met'}")
            print(f"  • Ollama server: {'✅ Connected' if status.get('ollama_connected') else '❌ Not connected'}")
            print(f"  • Qwen model: {'✅ Available' if status.get('model_available') else '❌ Not available'}")
            print(f"  • PDF document: {'✅ Loaded' if status.get('pdf_loaded') else '❌ Not loaded'}")
            print(f"  • Vector DB: {'✅ Ready' if status.get('vectordb_ready') else '❌ Not ready'}")
            
            # Configuration info
            print(f"\\n{Fore.CYAN}⚙️ Configuration:{Style.RESET_ALL}")
            print(f"  • Model: {self.config.model_name}")
            print(f"  • Config file: {self.config.config_path}")
            print(f"  • Cache directory: {self.config.cache_dir}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error during status check: {str(e)}{Style.RESET_ALL}")

    def edit_config(self):
        """Edit configuration file"""
        config_path = self.config.config_path
        
        print(f"{Fore.BLUE}⚙️ Configuration file path: {config_path}{Style.RESET_ALL}")
        
        # Create config file if not exists
        if not os.path.exists(config_path):
            self.config.save_config()
            print(f"{Fore.GREEN}✅ Default configuration file created.{Style.RESET_ALL}")
        
        # Open with system default editor
        try:
            os.system(f"{'notepad' if os.name == 'nt' else 'nano'} {config_path}")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to edit configuration file: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Please edit manually: {config_path}{Style.RESET_ALL}")

    def run(self):
        """Main execution function"""
        parser = argparse.ArgumentParser(
            description='DR-Bladder-CLI - Bladder Cancer EAU Guidelines AI Agent',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add command arguments
        parser.add_argument('command', nargs='?', choices=['query', 'chat', 'init', 'status', 'config'], 
                          help='Command to execute')
        parser.add_argument('question', nargs='?', help='Question content (for query command)')
        parser.add_argument('--model', '-m', default='qwen2.5:0.5b', help='Model to use (default: qwen2.5:0.5b)')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
        parser.add_argument('--config', '-c', help='Configuration file path')
        
        args = parser.parse_args()
        
        # Update configuration
        if args.config:
            self.config.config_path = args.config
        if args.model:
            self.config.model_name = args.model
        if args.verbose:
            self.config.verbose = True
        
        # Execute command
        if args.command == 'query':
            if not args.question:
                print(f"{Fore.RED}❌ Please provide a question.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}💡 Example: dr-bladder query \\"What are BCG side effects?\\"{Style.RESET_ALL}")
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
            # Show help if no command
            self.print_help()

if __name__ == '__main__':
    cli = BladderCLI()
    cli.run()