#!/usr/bin/env node

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const chalk = require('chalk');
const ora = require('ora');

class SystemInitializer {
    constructor() {
        this.pythonDir = path.join(__dirname, '..', 'python');
        this.requirementsPath = path.join(this.pythonDir, 'requirements.txt');
        this.configPath = path.join(os.homedir(), '.dr-bladder', 'config.json');
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

    warn(message) {
        console.log(chalk.yellow('⚠️'), message);
    }

    async init() {
        console.log(chalk.cyan('🚀 DR-Bladder-CLI 초기화 시작'));
        console.log('='.repeat(60));

        try {
            // 1. 시스템 요구사항 확인
            const systemOk = await this.checkSystemRequirements();
            if (!systemOk) {
                this.error('시스템 요구사항을 만족하지 않습니다.');
                process.exit(1);
            }

            // 2. Python 환경 확인
            const pythonCmd = await this.checkPython();
            if (!pythonCmd) {
                this.error('Python을 찾을 수 없습니다.');
                process.exit(1);
            }

            // 3. Python 의존성 설치
            const depsOk = await this.installPythonDependencies(pythonCmd);
            if (!depsOk) {
                this.error('Python 의존성 설치에 실패했습니다.');
                process.exit(1);
            }

            // 4. Ollama 확인 및 설치 안내
            const ollamaOk = await this.checkOllama();
            if (!ollamaOk) {
                this.showOllamaInstallInstructions();
                this.warn('Ollama 설치 후 다시 실행해주세요.');
                return;
            }

            // 5. 모델 설치
            const modelOk = await this.installModels();
            if (!modelOk) {
                this.error('모델 설치에 실패했습니다.');
                process.exit(1);
            }

            // 6. 설정 파일 생성
            await this.createConfigFile();

            // 7. PDF 파일 확인
            await this.checkPdfFile();

            // 8. 초기화 테스트
            await this.runInitializationTest(pythonCmd);

            console.log('\n' + '='.repeat(60));
            this.success('DR-Bladder-CLI 초기화 완료!');
            console.log('\n' + chalk.yellow('🎉 사용 방법:'));
            console.log('  dr-bladder query "BCG 치료의 부작용은?"');
            console.log('  dr-bladder chat');
            console.log('  dr-bladder status');
            console.log('');

        } catch (error) {
            this.error(`초기화 중 오류 발생: ${error.message}`);
            process.exit(1);
        }
    }

    async checkSystemRequirements() {
        const spinner = ora('시스템 요구사항 확인 중...').start();
        
        try {
            // Node.js 버전 확인
            const nodeVersion = process.version;
            const majorVersion = parseInt(nodeVersion.substring(1).split('.')[0]);
            
            if (majorVersion < 14) {
                spinner.fail(`Node.js 14 이상이 필요합니다. 현재: ${nodeVersion}`);
                return false;
            }
            
            // 운영체제 확인
            const platform = os.platform();
            const supportedPlatforms = ['win32', 'darwin', 'linux'];
            
            if (!supportedPlatforms.includes(platform)) {
                spinner.fail(`지원되지 않는 운영체제: ${platform}`);
                return false;
            }
            
            // 메모리 확인 (최소 4GB)
            const totalMemory = os.totalmem();
            const totalMemoryGB = totalMemory / (1024 * 1024 * 1024);
            
            if (totalMemoryGB < 4) {
                spinner.fail(`메모리 부족: ${totalMemoryGB.toFixed(1)}GB (최소 4GB 필요)`);
                return false;
            }
            
            spinner.succeed('시스템 요구사항 확인 완료');
            return true;
            
        } catch (error) {
            spinner.fail(`시스템 요구사항 확인 실패: ${error.message}`);
            return false;
        }
    }

    async checkPython() {
        const spinner = ora('Python 환경 확인 중...').start();
        
        return new Promise((resolve) => {
            const pythonCommands = ['python3', 'python'];
            
            const checkPython = (cmd) => {
                exec(`${cmd} --version`, (error, stdout, stderr) => {
                    if (error) {
                        return checkNextPython();
                    }
                    
                    const version = stdout.match(/Python (\d+\.\d+)/);
                    if (version) {
                        const majorMinor = version[1].split('.');
                        const major = parseInt(majorMinor[0]);
                        const minor = parseInt(majorMinor[1]);
                        
                        if (major === 3 && minor >= 8) {
                            spinner.succeed(`Python ${version[1]} 확인됨`);
                            resolve(cmd);
                        } else {
                            spinner.fail(`Python 3.8 이상이 필요합니다. 현재: ${version[1]}`);
                            resolve(null);
                        }
                    } else {
                        checkNextPython();
                    }
                });
            };
            
            let currentIndex = 0;
            const checkNextPython = () => {
                if (currentIndex >= pythonCommands.length) {
                    spinner.fail('Python을 찾을 수 없습니다.');
                    resolve(null);
                    return;
                }
                
                checkPython(pythonCommands[currentIndex]);
                currentIndex++;
            };
            
            checkNextPython();
        });
    }

    async installPythonDependencies(pythonCmd) {
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
                if (code === 0) {
                    spinner.succeed('Python 의존성 패키지 설치 완료');
                    resolve(true);
                } else {
                    spinner.fail('Python 의존성 패키지 설치 실패');
                    console.log(output);
                    resolve(false);
                }
            });
        });
    }

    async checkOllama() {
        const spinner = ora('Ollama 설치 확인 중...').start();
        
        return new Promise((resolve) => {
            exec('ollama --version', (error, stdout, stderr) => {
                if (error) {
                    spinner.fail('Ollama가 설치되지 않았습니다.');
                    resolve(false);
                } else {
                    spinner.succeed(`Ollama 확인됨: ${stdout.trim()}`);
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

    async installModels() {
        const spinner = ora('Qwen2.5-0.5B 모델 다운로드 중... (약 400MB)').start();
        
        return new Promise((resolve) => {
            const ollama = spawn('ollama', ['pull', 'qwen2.5:0.5b'], {
                stdio: 'pipe'
            });

            let output = '';
            ollama.stdout.on('data', (data) => {
                output += data.toString();
                // 진행률 표시
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
                if (code === 0) {
                    spinner.succeed('Qwen2.5-0.5B 모델 설치 완료');
                    resolve(true);
                } else {
                    spinner.fail('Qwen 모델 설치 실패');
                    console.log(output);
                    resolve(false);
                }
            });
        });
    }

    async createConfigFile() {
        const spinner = ora('설정 파일 생성 중...').start();
        
        try {
            const configDir = path.dirname(this.configPath);
            if (!fs.existsSync(configDir)) {
                fs.mkdirSync(configDir, { recursive: true });
            }

            const defaultConfig = {
                model_name: "qwen2.5:0.5b",
                ollama_host: "http://localhost:11434",
                embedding_model: "paraphrase-multilingual-MiniLM-L12-v2",
                top_k: 3,
                temperature: 0.7,
                max_tokens: 1000,
                language: "ko",
                verbose: false
            };

            fs.writeFileSync(this.configPath, JSON.stringify(defaultConfig, null, 2));
            spinner.succeed('설정 파일 생성 완료');
            
        } catch (error) {
            spinner.fail(`설정 파일 생성 실패: ${error.message}`);
        }
    }

    async checkPdfFile() {
        const spinner = ora('PDF 파일 확인 중...').start();
        
        const pdfPath = path.join(this.pythonDir, 'files', 'EAU-Guidelines-on-Non-muscle-invasive-Bladder-Cancer-2025.pdf');
        const sourcePdf = path.join(__dirname, '..', 'files', 'EAU-Guidelines-on-Non-muscle-invasive-Bladder-Cancer-2025.pdf');
        
        try {
            // 타겟 디렉토리 생성
            const filesDir = path.dirname(pdfPath);
            if (!fs.existsSync(filesDir)) {
                fs.mkdirSync(filesDir, { recursive: true });
            }

            // PDF 파일 복사
            if (fs.existsSync(sourcePdf)) {
                fs.copyFileSync(sourcePdf, pdfPath);
                spinner.succeed('PDF 파일 확인 완료');
            } else {
                spinner.warn('PDF 파일을 찾을 수 없습니다. 수동으로 추가해주세요.');
                console.log(`파일 경로: ${pdfPath}`);
            }
            
        } catch (error) {
            spinner.fail(`PDF 파일 처리 실패: ${error.message}`);
        }
    }

    async runInitializationTest(pythonCmd) {
        const spinner = ora('시스템 초기화 테스트 중...').start();
        
        return new Promise((resolve) => {
            const testScript = path.join(this.pythonDir, 'cli.py');
            const test = spawn(pythonCmd, [testScript, 'status'], {
                stdio: 'pipe',
                cwd: this.pythonDir
            });

            let output = '';
            test.stdout.on('data', (data) => {
                output += data.toString();
            });

            test.stderr.on('data', (data) => {
                output += data.toString();
            });

            test.on('close', (code) => {
                if (code === 0) {
                    spinner.succeed('시스템 초기화 테스트 완료');
                    resolve(true);
                } else {
                    spinner.fail('시스템 초기화 테스트 실패');
                    console.log(output);
                    resolve(false);
                }
            });
        });
    }
}

// 스크립트 실행
if (require.main === module) {
    const initializer = new SystemInitializer();
    initializer.init().catch(error => {
        console.error('초기화 중 오류:', error);
        process.exit(1);
    });
}

module.exports = SystemInitializer;