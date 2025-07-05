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
        console.log(chalk.blue('[DR-Bladder-CLI]'), message);
    }

    error(message) {
        console.error(chalk.red('[ERROR]'), message);
    }

    success(message) {
        console.log(chalk.green('[SUCCESS]'), message);
    }

    async checkPython() {
        return new Promise((resolve) => {
            const checkPythonVersion = (cmd, output) => {
                const version = output.match(/Python (\d+)\.(\d+)\.?(\d*)/);
                if (version) {
                    const major = parseInt(version[1]);
                    const minor = parseInt(version[2]);
                    const versionString = version[1] + '.' + version[2] + (version[3] ? '.' + version[3] : '');
                    
                    // Python 3.8 이상 확인
                    if (major > 3 || (major === 3 && minor >= 8)) {
                        this.success(`Python ${versionString} detected`);
                        return cmd.split(' ')[0];
                    } else {
                        this.error(`Python 3.8+ required. Current: ${versionString}`);
                        return false;
                    }
                } else {
                    this.error(`Cannot parse Python version: "${output}"`);
                    return false;
                }
            };

            exec('python3 --version', (error, stdout, stderr) => {
                const output = stdout || stderr;
                if (!error && output) {
                    const result = checkPythonVersion('python3 --version', output);
                    if (result) {
                        resolve(result);
                        return;
                    }
                }
                
                exec('python --version', (error2, stdout2, stderr2) => {
                    const output2 = stdout2 || stderr2;
                    if (!error2 && output2) {
                        const result = checkPythonVersion('python --version', output2);
                        resolve(result || false);
                    } else {
                        this.error('Python not found.');
                        this.log('Please install Python 3.8+: https://python.org');
                        resolve(false);
                    }
                });
            });
        });
    }

    async installPythonDeps(pythonCmd) {
        const spinner = ora('Installing Python dependencies...').start();
        
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
                    this.success('Python dependencies installed');
                    resolve(true);
                } else {
                    this.error('Python dependencies installation failed');
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
        console.log('\n' + chalk.cyan('Ollama installation required!'));
        console.log('=' .repeat(50));
        
        const platform = os.platform();
        
        if (platform === 'linux' || platform === 'darwin') {
            console.log(chalk.yellow('Linux/macOS installation:'));
            console.log('');
            console.log('1. Run in terminal:');
            console.log(chalk.green('   curl -fsSL https://ollama.ai/install.sh | sh'));
            console.log('');
            console.log('2. Start service:');
            console.log(chalk.green('   ollama serve &'));
        } else if (platform === 'win32') {
            console.log(chalk.yellow('Windows installation:'));
            console.log('');
            console.log('1. Download installer from:');
            console.log(chalk.blue('   https://ollama.ai/download'));
            console.log('');
            console.log('2. Run the installer');
            console.log('3. Service starts automatically');
        }
        
        console.log('');
        console.log(chalk.yellow('After installation:'));
        console.log('   dr-bladder init  # Run initialization');
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
        console.log(chalk.yellow('DR-Bladder-CLI Installation Started'));
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
        this.success('DR-Bladder-CLI basic installation completed!');
        
        if (!ollamaInstalled) {
            console.log('\n' + chalk.yellow('Next steps:'));
            console.log('1. Install Ollama following the instructions above');
            console.log('2. Install globally: ' + chalk.green('npm install -g .'));
            console.log('3. After installation: ' + chalk.green('dr-bladder init'));
        } else {
            console.log('\n' + chalk.yellow('Next steps:'));
            console.log('1. Install globally: ' + chalk.green('npm install -g .'));
            console.log('2. Initialize: ' + chalk.green('dr-bladder init'));
        }
        
        console.log('\n' + chalk.yellow('Usage after global install:'));
        console.log('dr-bladder query "What are BCG side effects?"');
        console.log('dr-bladder chat  # Interactive mode');
        console.log('dr-bladder status  # Check system status');
        console.log('\n' + chalk.yellow('Or use with npx:'));
        console.log('npx dr-bladder init');
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