# 🏥 DR-Bladder-CLI - 방광암 EAU 가이드라인 AI Agent

이 프로젝트는 방광암 EAU(유럽 비뇨기과 학회) 가이드라인을 기반으로 하는 독립 CLI AI Agent입니다. Ollama + Qwen2.5 모델을 활용하여 사용자가 방광암에 대해 자연어(한국어/영어)로 질문하면, 최신 가이드라인 PDF 문서에서 관련 정보를 찾아 AI가 생성한 답변을 제공합니다.

## 🚀 주요 기능

- **🤖 AI 기반 답변 생성**: Ollama + Qwen2.5 모델을 통한 지능적인 답변 생성
- **📚 최신 의학 정보**: 2025년 EAU 방광암 가이드라인 PDF에서 직접 정보를 검색
- **🗣️ 자연어 질문 답변**: "BCG 치료의 부작용은 무엇인가요?"와 같은 자연스러운 질문 처리
- **🌍 다국어 지원**: 한국어 질문을 영문 가이드라인에서 검색 가능
- **⚡ GPU 가속 지원**: NVIDIA GPU가 있는 경우 자동으로 GPU 활용
- **💬 대화형 모드**: 연속적인 질문과 답변이 가능한 채팅 인터페이스
- **🔧 간편한 설치**: 원클릭 설치 시스템으로 복잡한 설정 없이 바로 사용

## 📋 시스템 요구사항

- **운영체제**: Windows, macOS, Linux
- **Node.js**: 14.0 이상
- **Python**: 3.8 이상  
- **메모리**: 4GB 이상 권장
- **디스크**: 2GB 이상 여유 공간
- **Ollama**: AI 모델 실행을 위해 필요
- **GPU (선택 사항)**:
  - NVIDIA GPU (CUDA 지원)
  - VRAM 4GB 이상 권장

## 🚀 빠른 설치

### 1단계: 저장소 클론
```bash
git clone https://github.com/sk7944/DR_BLADDER_CLI.git
cd DR_BLADDER_CLI
```

### 2단계: 자동 설치
```bash
npm install  # Node.js 의존성 및 Python 패키지 자동 설치
```

### 3단계: Ollama 설치
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# https://ollama.ai/download 에서 설치 파일 다운로드
```

### 4단계: 전역 설치 (선택사항)
```bash
# 방법 1: 전역 설치 (dr-bladder 명령어 직접 사용)
npm install -g .

# 방법 2: npx 사용 (전역 설치 없이)
# npx dr-bladder 명령어로 사용
```

### 5단계: 초기화
```bash
# 전역 설치한 경우
dr-bladder init

# npx 사용하는 경우  
npx dr-bladder init
```

## 💬 사용 방법

### CLI 명령어

```bash
# 단일 질문
dr-bladder query "BCG 치료의 부작용은 무엇인가요?"

# 대화형 모드 (권장)
dr-bladder chat

# 시스템 상태 확인
dr-bladder status

# 설정 편집
dr-bladder config

# 도움말
dr-bladder --help
```

### 예시 질문들

**한국어:**
- "BCG 치료의 부작용은 무엇인가요?"
- "방광암의 재발 위험 요인에 대해 알려주세요"
- "TURBT 수술 후 관리 방법은?"
- "방광암 병기 분류에 대해 설명해주세요"

**English:**
- "What are the indications for BCG therapy?"
- "How is NMIBC risk stratification performed?"
- "What are the surveillance protocols for bladder cancer?"

### 대화형 모드 사용법

```bash
$ dr-bladder chat

🏥 DR-Bladder-CLI - 방광암 EAU 가이드라인 AI Agent
💬 대화형 모드 시작 (종료: 'quit', 'exit', 'q')

🤔 질문: BCG 치료의 부작용은?
🔍 답변 생성 중...

🏥 답변:
BCG 치료의 주요 부작용은 다음과 같습니다:
1. 국소 부작용: 배뇨 시 작열감, 빈뇨, 혈뇨
2. 전신 부작용: 발열, 피로감, 독감 유사 증상
3. 심각한 부작용: BCG균혈증 (드물지만 주의 필요)
...

🤔 질문: 
```

## 📁 프로젝트 구조

```
DR_BLADDER_CLI/
├── bin/
│   └── dr-bladder.js          # CLI 진입점
├── python/
│   ├── cli.py                 # Python CLI 메인
│   ├── bladder_agent.py       # 핵심 AI 에이전트
│   ├── config.py              # 설정 관리
│   ├── utils.py               # 유틸리티 함수들
│   ├── requirements.txt       # Python 의존성
│   └── files/                 # PDF 파일 저장소
│       └── EAU-Guidelines-*.pdf
├── src/
│   ├── install.js             # 자동 설치 시스템
│   ├── init.js                # 시스템 초기화
│   └── test.js                # 종합 테스트 시스템
├── files/
│   └── EAU-Guidelines-*.pdf   # 원본 PDF 파일
├── env/                       # Python 가상환경 (conda)
├── package.json               # Node.js 설정
└── README.md
```

## 🔧 핵심 구성 요소

| 파일 | 설명 |
|---|---|
| `bin/dr-bladder.js` | **CLI 진입점** - 모든 명령어의 시작점 |
| `python/cli.py` | **Python CLI 메인** - 실제 AI 기능을 담당 |
| `python/bladder_agent.py` | **AI 에이전트 핵심** - RAG + Ollama 통합 |
| `python/config.py` | **설정 관리** - 모든 설정을 관리 |
| `python/utils.py` | **유틸리티** - 시스템 검사, 로깅 등 |
| `src/install.js` | **자동 설치** - npm install 시 자동 실행 |
| `src/init.js` | **시스템 초기화** - Ollama, 모델 설치 |
| `src/test.js` | **종합 테스트** - 전체 시스템 검증 |

## 🔧 문제 해결

### 설치 관련 문제

1. **시스템 진단 실행**
   ```bash
   node src/test.js  # 종합 시스템 테스트
   ```

2. **상태 확인**
   ```bash
   dr-bladder status  # 현재 시스템 상태 확인
   ```

3. **재설치**
   ```bash
   npm install  # 의존성 재설치
   dr-bladder init  # 시스템 재초기화
   ```

### Ollama 관련 문제

1. **Ollama 서비스 확인**
   ```bash
   ollama --version  # Ollama 설치 확인
   ollama list       # 설치된 모델 목록
   ```

2. **Qwen 모델 수동 설치**
   ```bash
   ollama pull qwen2.5:0.5b
   ```

3. **Ollama 서비스 재시작**
   ```bash
   # Linux/macOS
   sudo systemctl restart ollama
   
   # 또는 직접 실행
   ollama serve
   ```

### 일반적인 오류

1. **"Python을 찾을 수 없습니다"**
   - Python 3.8+ 설치 확인
   - PATH 환경 변수 설정 확인

2. **"Ollama 연결 실패"**
   - Ollama 서비스 실행 여부 확인
   - 포트 11434 사용 가능 여부 확인

3. **"PDF 파일을 찾을 수 없습니다"**
   - `files/` 디렉토리에 PDF 파일 존재 확인
   - 파일 권한 확인

4. **메모리 부족 오류**
   - GPU 메모리가 부족한 경우 배치 크기 조정
   - 설정 파일에서 `batch_size` 값 감소
   - CPU 사용으로 전환 (GPU 메모리 절약)

## 🛠️ 기술 스택

### 핵심 기술
- **🤖 Ollama + Qwen2.5-0.5B**: 경량 로컬 AI 모델 실행
- **🔍 RAG (Retrieval-Augmented Generation)**: 문서 기반 답변 생성
- **⚡ ChromaDB**: 고성능 벡터 데이터베이스
- **🧠 SentenceTransformers**: 다국어 텍스트 임베딩

### 개발 환경
- **🐍 Python 3.8+**: AI 백엔드
- **📦 Node.js 14+**: CLI 인터페이스
- **🔥 PyTorch**: 딥러닝 프레임워크 (GPU/CPU)
- **📄 PyPDF2**: PDF 문서 처리

### 기타 라이브러리
- **🎨 Colorama**: 터미널 컬러 출력
- **📊 psutil**: 시스템 모니터링
- **⏱️ tqdm**: 진행률 표시
- **⚙️ argparse**: CLI 인터페이스

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## ⚠️ 면책 조항

이 프로젝트는 **정보 제공을 목적**으로 하며, **의학적 조언을 대체할 수 없습니다**. 

⚠️ **중요**: 모든 치료 결정은 반드시 **전문 의료인과 상의**하시기 바랍니다.

## 🤝 기여하기

프로젝트 개선을 위한 기여를 환영합니다!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 지원

문제가 발생하면 다음을 확인해주세요:

1. **GitHub Issues**: 버그 리포트 및 기능 요청
2. **시스템 테스트**: `node src/test.js`
3. **상태 확인**: `dr-bladder status`

---

**🏥 DR-Bladder-CLI** - 의료진의 더 나은 진료를 위한 AI 도구