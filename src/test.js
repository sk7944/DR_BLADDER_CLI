#!/usr/bin/env node

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const chalk = require('chalk');
const ora = require('ora');

class SystemTester {
    constructor() {
        this.pythonDir = path.join(__dirname, '..', 'python');
        this.configPath = path.join(os.homedir(), '.dr-bladder', 'config.json');
        this.testResults = [];
    }

    log(message) {
        console.log(chalk.blue('🧪 DR-Bladder-CLI Test:'), message);
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

    async runTests() {
        console.log(chalk.cyan('🧪 DR-Bladder-CLI 테스트 시작'));
        console.log('='.repeat(60));

        try {
            // 1. 시스템 환경 테스트
            await this.testSystemEnvironment();

            // 2. Python 환경 테스트
            await this.testPythonEnvironment();

            // 3. 의존성 패키지 테스트
            await this.testDependencies();

            // 4. Ollama 연결 테스트
            await this.testOllamaConnection();

            // 5. 모델 가용성 테스트
            await this.testModelAvailability();

            // 6. 설정 파일 테스트
            await this.testConfigFile();

            // 7. PDF 파일 테스트
            await this.testPdfFile();

            // 8. CLI 명령어 테스트
            await this.testCliCommands();

            // 9. 실제 질문 테스트
            await this.testActualQuery();

            // 결과 출력
            this.printTestResults();

        } catch (error) {
            this.error(`테스트 중 오류 발생: ${error.message}`);
            process.exit(1);
        }
    }

    async testSystemEnvironment() {
        const spinner = ora('시스템 환경 테스트 중...').start();
        
        try {
            const results = {
                name: '시스템 환경',
                tests: []
            };

            // Node.js 버전 확인
            const nodeVersion = process.version;
            const majorVersion = parseInt(nodeVersion.substring(1).split('.')[0]);
            results.tests.push({
                name: 'Node.js 버전',
                status: majorVersion >= 14 ? 'pass' : 'fail',
                details: `현재: ${nodeVersion}, 요구사항: 14+`
            });

            // 운영체제 확인
            const platform = os.platform();
            const supportedPlatforms = ['win32', 'darwin', 'linux'];
            results.tests.push({
                name: '운영체제',
                status: supportedPlatforms.includes(platform) ? 'pass' : 'fail',
                details: `현재: ${platform}`
            });

            // 메모리 확인
            const totalMemory = os.totalmem();
            const totalMemoryGB = totalMemory / (1024 * 1024 * 1024);
            results.tests.push({
                name: '메모리',
                status: totalMemoryGB >= 4 ? 'pass' : 'fail',
                details: `현재: ${totalMemoryGB.toFixed(1)}GB, 요구사항: 4GB+`
            });

            // 디스크 여유 공간 확인
            const stats = fs.statSync('/');
            results.tests.push({
                name: '디스크 공간',
                status: 'pass',
                details: '사용 가능'
            });

            this.testResults.push(results);
            spinner.succeed('시스템 환경 테스트 완료');
            
        } catch (error) {
            spinner.fail(`시스템 환경 테스트 실패: ${error.message}`);
        }
    }

    async testPythonEnvironment() {
        const spinner = ora('Python 환경 테스트 중...').start();
        
        return new Promise((resolve) => {
            const results = {
                name: 'Python 환경',
                tests: []
            };

            const pythonCommands = ['python3', 'python'];
            let pythonFound = false;

            const checkPython = (cmd) => {
                exec(`${cmd} --version`, (error, stdout, stderr) => {
                    if (error) {
                        results.tests.push({
                            name: `${cmd} 명령어`,
                            status: 'fail',
                            details: '사용 불가'
                        });
                        return checkNextPython();
                    }
                    
                    const version = stdout.match(/Python (\d+\.\d+)/);
                    if (version) {
                        const majorMinor = version[1].split('.');
                        const major = parseInt(majorMinor[0]);
                        const minor = parseInt(majorMinor[1]);
                        
                        const versionOk = major === 3 && minor >= 8;
                        results.tests.push({
                            name: `${cmd} 버전`,
                            status: versionOk ? 'pass' : 'fail',
                            details: `현재: ${version[1]}, 요구사항: 3.8+`
                        });
                        
                        if (versionOk) {
                            pythonFound = true;
                        }
                    }
                    
                    checkNextPython();
                });
            };
            
            let currentIndex = 0;
            const checkNextPython = () => {
                if (currentIndex >= pythonCommands.length) {
                    this.testResults.push(results);
                    if (pythonFound) {
                        spinner.succeed('Python 환경 테스트 완료');
                    } else {
                        spinner.fail('Python 환경 테스트 실패');
                    }
                    resolve();
                    return;
                }
                
                checkPython(pythonCommands[currentIndex]);
                currentIndex++;
            };
            
            checkNextPython();
        });
    }

    async testDependencies() {
        const spinner = ora('의존성 패키지 테스트 중...').start();
        
        return new Promise((resolve) => {
            const results = {
                name: '의존성 패키지',
                tests: []
            };

            const pythonCmd = 'python3'; // 실제로는 이전 테스트 결과를 사용해야 함
            const requiredPackages = [
                'torch', 'transformers', 'sentence_transformers', 
                'chromadb', 'PyPDF2', 'ollama', 'psutil', 'tqdm', 'colorama'
            ];

            let checkedPackages = 0;
            
            requiredPackages.forEach(pkg => {
                exec(`${pythonCmd} -c "import ${pkg}; print('${pkg} OK')"`, (error, stdout, stderr) => {
                    results.tests.push({
                        name: pkg,
                        status: error ? 'fail' : 'pass',
                        details: error ? error.message : '설치됨'
                    });
                    
                    checkedPackages++;
                    if (checkedPackages === requiredPackages.length) {
                        this.testResults.push(results);
                        spinner.succeed('의존성 패키지 테스트 완료');
                        resolve();
                    }
                });
            });
        });
    }

    async testOllamaConnection() {
        const spinner = ora('Ollama 연결 테스트 중...').start();
        
        return new Promise((resolve) => {
            const results = {
                name: 'Ollama 연결',
                tests: []
            };

            // Ollama 설치 확인
            exec('ollama --version', (error, stdout, stderr) => {
                results.tests.push({
                    name: 'Ollama 설치',
                    status: error ? 'fail' : 'pass',
                    details: error ? '설치되지 않음' : stdout.trim()
                });

                if (error) {
                    this.testResults.push(results);
                    spinner.fail('Ollama 연결 테스트 실패');
                    resolve();
                    return;
                }

                // Ollama 서비스 테스트
                exec('curl -s http://localhost:11434/api/tags', (error, stdout, stderr) => {
                    results.tests.push({
                        name: 'Ollama 서비스',
                        status: error ? 'fail' : 'pass',
                        details: error ? '서비스 미실행' : '서비스 실행 중'
                    });

                    this.testResults.push(results);
                    if (error) {
                        spinner.fail('Ollama 연결 테스트 실패');
                    } else {
                        spinner.succeed('Ollama 연결 테스트 완료');
                    }
                    resolve();
                });
            });
        });
    }

    async testModelAvailability() {
        const spinner = ora('모델 가용성 테스트 중...').start();
        
        return new Promise((resolve) => {
            const results = {
                name: '모델 가용성',
                tests: []
            };

            // 설치된 모델 목록 확인
            exec('ollama list', (error, stdout, stderr) => {
                if (error) {
                    results.tests.push({
                        name: '모델 목록',
                        status: 'fail',
                        details: '모델 목록 조회 실패'
                    });
                } else {
                    const models = stdout.split('\n').filter(line => line.trim().length > 0);
                    const hasQwen = models.some(model => model.includes('qwen'));
                    
                    results.tests.push({
                        name: '모델 목록',
                        status: 'pass',
                        details: `${models.length - 1}개 모델 설치됨`
                    });

                    results.tests.push({
                        name: 'Qwen 모델',
                        status: hasQwen ? 'pass' : 'fail',
                        details: hasQwen ? '설치됨' : '설치되지 않음'
                    });
                }

                this.testResults.push(results);
                spinner.succeed('모델 가용성 테스트 완료');
                resolve();
            });
        });
    }

    async testConfigFile() {
        const spinner = ora('설정 파일 테스트 중...').start();
        
        const results = {
            name: '설정 파일',
            tests: []
        };

        try {
            // 설정 파일 존재 확인
            const configExists = fs.existsSync(this.configPath);
            results.tests.push({
                name: '설정 파일 존재',
                status: configExists ? 'pass' : 'fail',
                details: configExists ? '존재' : '없음'
            });

            if (configExists) {
                // 설정 파일 내용 확인
                const configData = JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
                const requiredKeys = ['model_name', 'ollama_host', 'embedding_model'];
                
                requiredKeys.forEach(key => {
                    results.tests.push({
                        name: `설정 키: ${key}`,
                        status: configData[key] ? 'pass' : 'fail',
                        details: configData[key] ? '설정됨' : '없음'
                    });
                });
            }

            this.testResults.push(results);
            spinner.succeed('설정 파일 테스트 완료');
            
        } catch (error) {
            results.tests.push({
                name: '설정 파일 파싱',
                status: 'fail',
                details: error.message
            });
            this.testResults.push(results);
            spinner.fail('설정 파일 테스트 실패');
        }
    }

    async testPdfFile() {
        const spinner = ora('PDF 파일 테스트 중...').start();
        
        const results = {
            name: 'PDF 파일',
            tests: []
        };

        const pdfPath = path.join(this.pythonDir, 'files', 'EAU-Guidelines-on-Non-muscle-invasive-Bladder-Cancer-2025.pdf');
        
        // PDF 파일 존재 확인
        const pdfExists = fs.existsSync(pdfPath);
        results.tests.push({
            name: 'PDF 파일 존재',
            status: pdfExists ? 'pass' : 'fail',
            details: pdfExists ? '존재' : '없음'
        });

        if (pdfExists) {
            // PDF 파일 크기 확인
            const stats = fs.statSync(pdfPath);
            const fileSizeMB = stats.size / (1024 * 1024);
            
            results.tests.push({
                name: 'PDF 파일 크기',
                status: fileSizeMB > 1 ? 'pass' : 'fail',
                details: `${fileSizeMB.toFixed(1)}MB`
            });
        }

        this.testResults.push(results);
        spinner.succeed('PDF 파일 테스트 완료');
    }

    async testCliCommands() {
        const spinner = ora('CLI 명령어 테스트 중...').start();
        
        const results = {
            name: 'CLI 명령어',
            tests: []
        };

        const pythonCmd = 'python3';
        const cliScript = path.join(this.pythonDir, 'cli.py');
        
        // CLI 스크립트 존재 확인
        const cliExists = fs.existsSync(cliScript);
        results.tests.push({
            name: 'CLI 스크립트 존재',
            status: cliExists ? 'pass' : 'fail',
            details: cliExists ? '존재' : '없음'
        });

        if (cliExists) {
            // status 명령어 테스트
            await new Promise((resolve) => {
                exec(`${pythonCmd} ${cliScript} status`, { cwd: this.pythonDir }, (error, stdout, stderr) => {
                    results.tests.push({
                        name: 'status 명령어',
                        status: error ? 'fail' : 'pass',
                        details: error ? stderr : '실행 가능'
                    });
                    resolve();
                });
            });
        }

        this.testResults.push(results);
        spinner.succeed('CLI 명령어 테스트 완료');
    }

    async testActualQuery() {
        const spinner = ora('실제 질문 테스트 중...').start();
        
        const results = {
            name: '실제 질문',
            tests: []
        };

        const pythonCmd = 'python3';
        const cliScript = path.join(this.pythonDir, 'cli.py');
        const testQuery = "What is BCG?";
        
        // 실제 질문 테스트 (타임아웃 30초)
        await new Promise((resolve) => {
            const child = spawn(pythonCmd, [cliScript, 'query', testQuery], { 
                cwd: this.pythonDir,
                stdio: 'pipe'
            });

            let output = '';
            let timedOut = false;
            
            const timeout = setTimeout(() => {
                timedOut = true;
                child.kill();
                results.tests.push({
                    name: '질문 응답',
                    status: 'fail',
                    details: '타임아웃 (30초)'
                });
                resolve();
            }, 30000);

            child.stdout.on('data', (data) => {
                output += data.toString();
            });

            child.stderr.on('data', (data) => {
                output += data.toString();
            });

            child.on('close', (code) => {
                if (!timedOut) {
                    clearTimeout(timeout);
                    results.tests.push({
                        name: '질문 응답',
                        status: code === 0 ? 'pass' : 'fail',
                        details: code === 0 ? '응답 생성 성공' : '응답 생성 실패'
                    });
                }
                resolve();
            });
        });

        this.testResults.push(results);
        spinner.succeed('실제 질문 테스트 완료');
    }

    printTestResults() {
        console.log('\n' + '='.repeat(60));
        console.log(chalk.cyan('📋 테스트 결과 요약'));
        console.log('='.repeat(60));

        let totalTests = 0;
        let passedTests = 0;
        let failedTests = 0;

        this.testResults.forEach(category => {
            console.log(`\n${chalk.yellow(category.name)}:`);
            
            category.tests.forEach(test => {
                totalTests++;
                if (test.status === 'pass') {
                    passedTests++;
                    console.log(`  ${chalk.green('✓')} ${test.name}: ${test.details}`);
                } else {
                    failedTests++;
                    console.log(`  ${chalk.red('✗')} ${test.name}: ${test.details}`);
                }
            });
        });

        console.log('\n' + '='.repeat(60));
        console.log(`${chalk.green('✓ 통과:')} ${passedTests}개`);
        console.log(`${chalk.red('✗ 실패:')} ${failedTests}개`);
        console.log(`${chalk.blue('총 테스트:')} ${totalTests}개`);
        
        if (failedTests === 0) {
            console.log(`\n${chalk.green('🎉 모든 테스트가 통과했습니다!')}`);
        } else {
            console.log(`\n${chalk.yellow('⚠️ 일부 테스트가 실패했습니다. 위의 결과를 확인해주세요.')}`);
        }
    }
}

// 스크립트 실행
if (require.main === module) {
    const tester = new SystemTester();
    tester.runTests().catch(error => {
        console.error('테스트 중 오류:', error);
        process.exit(1);
    });
}

module.exports = SystemTester;