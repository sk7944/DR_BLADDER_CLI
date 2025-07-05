#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Python 스크립트 경로
const pythonDir = path.join(__dirname, '..', 'python');
const cliScript = path.join(pythonDir, 'cli.py');

// Python 실행 파일 찾기
function findPython() {
    const pythonCommands = ['python3', 'python'];
    
    for (const cmd of pythonCommands) {
        try {
            const result = require('child_process').execSync(`${cmd} --version`, { stdio: 'pipe' });
            if (result) {
                return cmd;
            }
        } catch (error) {
            continue;
        }
    }
    
    console.error('❌ Python not found. Please install Python 3.8 or higher.');
    process.exit(1);
}

// 메인 실행 함수
function main() {
    const pythonCmd = findPython();
    const args = process.argv.slice(2);
    
    // Python CLI 실행
    const python = spawn(pythonCmd, [cliScript, ...args], {
        stdio: 'inherit',
        cwd: pythonDir
    });
    
    python.on('close', (code) => {
        process.exit(code);
    });
    
    python.on('error', (error) => {
        console.error(`❌ Python execution error: ${error.message}`);
        process.exit(1);
    });
}

// 도움말 표시
if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
🏥 DR-Bladder-CLI - Bladder Cancer EAU Guidelines AI Agent

Usage:
  dr-bladder query "What are BCG side effects?"     # Ask questions
  dr-bladder chat                                    # Interactive mode
  dr-bladder init                                    # Initial setup
  dr-bladder status                                  # Check status
  dr-bladder --help                                  # Show help

First run after installation:
  dr-bladder init    # Install Ollama and Qwen model
    `);
    process.exit(0);
}

// 실행
main();