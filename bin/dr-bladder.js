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
    
    console.error('❌ Python을 찾을 수 없습니다. Python 3.8 이상을 설치해주세요.');
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
        console.error(`❌ Python 실행 오류: ${error.message}`);
        process.exit(1);
    });
}

// 도움말 표시
if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
🏥 DR-Bladder-CLI - 방광암 EAU 가이드라인 AI Agent

사용법:
  dr-bladder query "BCG 치료의 부작용은?"     # 질문하기
  dr-bladder init                            # 초기 설정
  dr-bladder status                          # 상태 확인
  dr-bladder --help                          # 도움말

설치 후 첫 실행:
  dr-bladder init    # Ollama 및 Qwen 모델 설치
    `);
    process.exit(0);
}

// 실행
main();