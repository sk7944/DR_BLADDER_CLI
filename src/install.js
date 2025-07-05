#!/usr/bin/env node

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const chalk = require('chalk');
const ora = require('ora');

class Installer {
    constructor() {
        this.pythonDir = path.join(__dirname, '..', 'python');
        this.requirementsPath = path.join(this.pythonDir, 'requirements.txt');
    }

    log(message) {
        console.log(chalk.blue('🏥 DR-Bladder-CLI:'), message);
    }

    error(message) {
        console.error(chalk.red('❌ 오류:'), message);
    }

    success(message) {
        console.log(chalk.green('✅'), message);
    }

    async checkPython() {
        return new Promise((resolve) => {
            exec('python3 --version', (error, stdout) => {
                if (error) {
                    exec('python --version', (error2, stdout2) => {
                        if (error2) {
                            this.error('Python을 찾을 수 없습니다.');
                            this.log('Python 3.8 이상을 설치해주세요: https://python.org');
                            resolve(false);
                        } else {
                            const version = stdout2.match(/Python (\d+\.\d+)/);
                            if (version && parseFloat(version[1]) >= 3.8) {
                                this.success(`Python ${version[1]} 확인됨`);
                                resolve('python');
                            } else {
                                this.error('Python 3.8 이상이 필요합니다.');
                                resolve(false);
                            }
                        }
                    });
                } else {
                    const version = stdout.match(/Python (\d+\.\d+)/);
                    if (version && parseFloat(version[1]) >= 3.8) {
                        this.success(`Python ${version[1]} 확인됨`);
                        resolve('python3');
                    } else {
                        this.error('Python 3.8 이상이 필요합니다.');
                        resolve(false);
                    }
                }
            });
        });
    }

    async installPythonDeps(pythonCmd) {
        const spinner = ora('Python 의존성 패키지 설치 중...').start();
        
        return new Promise((resolve) => {
            const pip = spawn(pythonCmd, ['-m', 'pip', 'install', '-r', this.requirementsPath], {
                stdio: 'pipe'
            });

            let output = '';
            pip.stdout.on('data', (data) => {
                output += data.toString();
            });

            pip.stderr.on('data', (data) => {
                output += data.toString();
            });

            pip.on('close', (code) => {
                spinner.stop();
                if (code === 0) {
                    this.success('Python 패키지 설치 완료');
                    resolve(true);
                } else {
                    this.error('Python 패키지 설치 실패');
                    console.log(output);
                    resolve(false);
                }
            });
        });
    }

    async checkOllama() {
        return new Promise((resolve) => {
            exec('ollama --version', (error, stdout) => {
                if (error) {
                    this.showOllamaInstallInstructions();
                    resolve(false);
                } else {
                    this.success(`Ollama 확인됨: ${stdout.trim()}`);
                    resolve(true);
                }
            });
        });
    }

    showOllamaInstallInstructions() {
        console.log('\n' + chalk.cyan('🤖 Ollama 설치가 필요합니다!'));
        console.log('=' .repeat(50));
        
        const platform = os.platform();
        
        if (platform === 'linux' || platform === 'darwin') {
            console.log(chalk.yellow('📋 Linux/macOS 설치 방법:'));
            console.log('');
            console.log('1. 터미널에서 다음 명령어 실행:');
            console.log(chalk.green('   curl -fsSL https://ollama.ai/install.sh | sh'));
            console.log('');
            console.log('2. 설치 완료 후 서비스 시작:');
            console.log(chalk.green('   ollama serve &'));
        } else if (platform === 'win32') {
            console.log(chalk.yellow('📋 Windows 설치 방법:'));
            console.log('');
            console.log('1. 다음 링크에서 설치 파일 다운로드:');
            console.log(chalk.blue('   https://ollama.ai/download'));
            console.log('');
            console.log('2. 다운로드한 설치 파일 실행');
            console.log('3. 설치 완료 후 자동으로 서비스 시작됨');
        }
        
        console.log('');
        console.log(chalk.yellow('💡 설치 완료 후:'));
        console.log('   dr-bladder init  # 이 명령어로 초기화 재시도');
        console.log('');
    }

    async installQwenModel() {
        const spinner = ora('Qwen2.5-0.5B 모델 다운로드 중... (약 400MB, 시간이 걸릴 수 있습니다)').start();
        
        return new Promise((resolve) => {
            const ollama = spawn('ollama', ['pull', 'qwen2.5:0.5b'], {
                stdio: 'pipe'
            });

            let output = '';
            ollama.stdout.on('data', (data) => {
                output += data.toString();
                // 진행률 업데이트
                const lines = output.split('\n');
                const lastLine = lines[lines.length - 2] || '';
                if (lastLine.includes('%')) {
                    spinner.text = `Qwen 모델 다운로드 중... ${lastLine}`;
                }
            });

            ollama.stderr.on('data', (data) => {
                output += data.toString();
            });

            ollama.on('close', (code) => {
                spinner.stop();
                if (code === 0) {
                    this.success('Qwen2.5-0.5B 모델 설치 완료');
                    resolve(true);
                } else {
                    this.error('Qwen 모델 설치 실패');
                    console.log(output);
                    resolve(false);
                }
            });
        });
    }

    async copyPdfFile() {
        const sourcePdf = path.join(__dirname, '..', 'files', 'EAU-Guidelines-on-Non-muscle-invasive-Bladder-Cancer-2025.pdf');
        const targetPdf = path.join(this.pythonDir, 'files', 'EAU-Guidelines-on-Non-muscle-invasive-Bladder-Cancer-2025.pdf');
        
        try {
            // python/files 디렉토리 생성
            const filesDir = path.dirname(targetPdf);
            if (!fs.existsSync(filesDir)) {
                fs.mkdirSync(filesDir, { recursive: true });
            }

            if (fs.existsSync(sourcePdf)) {
                fs.copyFileSync(sourcePdf, targetPdf);
                this.success('PDF 파일 복사 완료');
                return true;
            } else {
                this.error('PDF 파일을 찾을 수 없습니다: ' + sourcePdf);
                return false;
            }
        } catch (error) {
            this.error('PDF 파일 복사 실패: ' + error.message);
            return false;
        }
    }

    async run() {
        console.log(chalk.yellow('🏥 DR-Bladder-CLI 설치 시작'));
        console.log('='.repeat(50));

        // 1. Python 확인
        const pythonCmd = await this.checkPython();
        if (!pythonCmd) {
            process.exit(1);
        }

        // 2. Python 의존성 설치
        const depsInstalled = await this.installPythonDeps(pythonCmd);
        if (!depsInstalled) {
            this.error('Python 의존성 설치 실패. 수동으로 실행해보세요:');
            this.log(`${pythonCmd} -m pip install -r ${this.requirementsPath}`);
            process.exit(1);
        }

        // 3. PDF 파일 복사
        await this.copyPdfFile();

        // 4. Ollama 확인 (설치 안내만)
        const ollamaInstalled = await this.checkOllama();
        
        console.log('\n' + '='.repeat(50));
        this.success('DR-Bladder-CLI 기본 설치 완료!');
        
        if (!ollamaInstalled) {
            console.log('\n' + chalk.yellow('⚠️  다음 단계:'));
            console.log('1. 위의 안내를 따라 Ollama를 설치하세요');
            console.log('2. 설치 완료 후: ' + chalk.green('dr-bladder init'));
        } else {
            console.log('\n' + chalk.yellow('🚀 다음 단계:'));
            console.log(chalk.green('dr-bladder init') + '  # Qwen 모델 다운로드 및 초기화');
        }
        
        console.log('\n' + chalk.yellow('💡 사용 방법:'));
        console.log('dr-bladder query "BCG 치료의 부작용은?"');
        console.log('dr-bladder chat  # 대화형 모드');
        console.log('dr-bladder status  # 시스템 상태 확인');
        console.log('');
    }
}

// 설치 실행
if (require.main === module) {
    const installer = new Installer();
    installer.run().catch(error => {
        console.error('설치 중 오류:', error);
        process.exit(1);
    });
}

module.exports = Installer;