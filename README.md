# DR-Bladder-CLI - Bladder Cancer EAU Guidelines AI Agent

This project is an independent CLI AI Agent based on bladder cancer EAU (European Association of Urology) guidelines. It utilizes Ollama + Qwen2.5-1.5B model to provide AI-generated answers by searching relevant information from the latest guideline PDF documents when users ask questions about bladder cancer in natural language. All responses are provided in English regardless of the input language.

## Key Features

- **AI-powered Answer Generation**: Intelligent answer generation through Ollama + Qwen2.5-1.5B model
- **Latest Medical Information**: Direct information retrieval from 2025 EAU Bladder Cancer Guidelines PDF
- **Natural Language Q&A**: Handles natural questions like "What are the side effects of BCG treatment?"
- **Multi-language Input**: Accepts questions in Korean or English, but always responds in English
- **GPU Acceleration**: Automatically utilizes NVIDIA GPU when available
- **Interactive Mode**: Chat interface enabling continuous questions and answers
- **Easy Installation**: One-click installation system with progress tracking for immediate use without complex setup

## System Requirements

- **Operating System**: Windows, macOS, Linux
- **Node.js**: 14.0 or higher
- **Python**: 3.8 or higher  
- **Memory**: 4GB or more recommended
- **Disk Space**: 3GB or more free space (for Qwen2.5-1.5B model)
- **Ollama**: Required for AI model execution
- **GPU (Optional)**:
  - NVIDIA GPU (CUDA support)
  - 4GB VRAM or more recommended

## Quick Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/sk7944/DR_BLADDER_CLI.git
cd DR_BLADDER_CLI
```

### Step 2: Automatic Installation
```bash
npm install  # Automatically installs Node.js dependencies and Python packages
```

### Step 3: Install Ollama
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download installer from https://ollama.ai/download
```

### Step 4: Global Installation (Optional)
```bash
# Method 1: Global installation (use dr-bladder command directly)
npm install -g .

# Method 2: Use npx (without global installation)
# Use with npx dr-bladder command
```

### Step 5: Initialize
```bash
# If globally installed
dr-bladder init

# If using npx
npx dr-bladder init
```

## Usage

### CLI Commands

```bash
# Single question
dr-bladder query "What are the side effects of BCG treatment?"

# Interactive mode (recommended)
dr-bladder chat

# Check system status
dr-bladder status

# Edit configuration
dr-bladder config

# Help
dr-bladder --help
```

### Example Questions

**Input Examples (Korean - responds in English):**
- "BCG 치료의 부작용은 무엇인가요?" → English response about BCG side effects
- "방광암의 재발 위험 요인에 대해 알려주세요" → English response about recurrence risk factors
- "TURBT 수술 후 관리 방법은?" → English response about post-TURBT management
- "방광암 병기 분류에 대해 설명해주세요" → English response about staging classification

**Input Examples (English):**
- "What are the indications for BCG therapy?"
- "How is NMIBC risk stratification performed?"
- "What are the surveillance protocols for bladder cancer?"

### Interactive Mode Usage

```bash
$ dr-bladder chat

DR-Bladder-CLI - Bladder Cancer EAU Guidelines AI Agent
Interactive mode started (Exit: 'quit', 'exit', 'q')

Question: What are the side effects of BCG?
Generating answer...

Answer:
The main side effects of BCG therapy include:
1. Local side effects: Burning sensation during urination, frequent urination, hematuria
2. Systemic side effects: Fever, fatigue, flu-like symptoms
3. Serious side effects: BCG sepsis (rare but requires attention)
...

Question: 
```

## Project Structure

```
DR_BLADDER_CLI/
├── bin/
│   └── dr-bladder.js          # CLI entry point
├── python/
│   ├── cli.py                 # Python CLI main
│   ├── bladder_agent.py       # Core AI agent
│   ├── config.py              # Configuration management
│   ├── utils.py               # Utility functions
│   ├── requirements.txt       # Python dependencies
│   └── files/                 # PDF file storage
│       └── EAU-Guidelines-*.pdf
├── src/
│   ├── install.js             # Automatic installation system
│   ├── init.js                # System initialization
│   └── test.js                # Comprehensive test system
├── files/
│   └── EAU-Guidelines-*.pdf   # Original PDF files
├── env/                       # Python virtual environment (conda)
├── package.json               # Node.js configuration
└── README.md
```

## Core Components

| File | Description |
|---|---|
| `bin/dr-bladder.js` | **CLI Entry Point** - Starting point for all commands |
| `python/cli.py` | **Python CLI Main** - Handles actual AI functionality |
| `python/bladder_agent.py` | **AI Agent Core** - RAG + Ollama integration |
| `python/config.py` | **Configuration Management** - Manages all settings |
| `python/utils.py` | **Utilities** - System checks, logging, etc. |
| `src/install.js` | **Automatic Installation** - Runs automatically on npm install |
| `src/init.js` | **System Initialization** - Ollama and model installation |
| `src/test.js` | **Comprehensive Testing** - Full system verification |

## Troubleshooting

### Installation Issues

1. **Run System Diagnostics**
   ```bash
   node src/test.js  # Comprehensive system test
   ```

2. **Check Status**
   ```bash
   dr-bladder status  # Check current system status
   ```

3. **Reinstall**
   ```bash
   npm install  # Reinstall dependencies
   dr-bladder init  # Reinitialize system
   ```

### Ollama Issues

1. **Check Ollama Service**
   ```bash
   ollama --version  # Check Ollama installation
   ollama list       # List installed models
   ```

2. **Manual Qwen Model Installation**
   ```bash
   ollama pull qwen2.5:1.5b
   ```

3. **Restart Ollama Service**
   ```bash
   # Linux/macOS
   sudo systemctl restart ollama
   
   # Or run directly
   ollama serve
   ```

### Common Errors

1. **"Python not found"**
   - Check Python 3.8+ installation
   - Verify PATH environment variable settings

2. **"Ollama connection failed"**
   - Check if Ollama service is running
   - Verify port 11434 availability

3. **"PDF file not found"**
   - Check PDF file existence in `files/` directory
   - Verify file permissions

4. **Out of memory error**
   - Adjust batch size if GPU memory is insufficient
   - Reduce `batch_size` value in configuration file
   - Switch to CPU usage (saves GPU memory)

## Tech Stack

### Core Technologies
- **Ollama + Qwen2.5-1.5B**: Enhanced local AI model execution with progress tracking
- **RAG (Retrieval-Augmented Generation)**: Document-based answer generation
- **ChromaDB**: High-performance vector database
- **SentenceTransformers**: Multilingual text embedding with improved encoding handling

### Development Environment
- **Python 3.8+**: AI backend
- **Node.js 14+**: CLI interface
- **PyTorch**: Deep learning framework (GPU/CPU)
- **PyPDF2**: PDF document processing

### Additional Libraries
- **Colorama**: Terminal color output
- **psutil**: System monitoring
- **tqdm**: Progress display
- **argparse**: CLI interface

## License

This project is distributed under the MIT License.

## Disclaimer

This project is intended for **informational purposes** and **cannot replace medical advice**. 

**Important**: All treatment decisions must be discussed with **qualified healthcare professionals**.

## Contributing

Contributions for project improvement are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Support

If you encounter issues, please check:

1. **GitHub Issues**: Bug reports and feature requests
2. **System Test**: `node src/test.js`
3. **Status Check**: `dr-bladder status`

---

**DR-Bladder-CLI** - AI tool for better healthcare by medical professionals