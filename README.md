# OpenNarrator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI Pipeline](https://github.com/yourusername/open_narrator/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/open_narrator/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/yourusername/open_narrator/pkgs/container/open_narrator)

> Self-hosted audio translation and audiobook creation service using Whisper, Claude, and Piper TTS

OpenNarrator is an open-source application that:
- **Translates audio files** (MP3) from English to multiple target languages while preserving natural narration quality
- **Creates audiobooks from text files** (TXT, PDF, EPUB, MOBI, DOCX, and more) in 30+ languages

Perfect for translating audiobooks, podcasts, educational content, or converting your ebooks into professionally narrated audiobooks.

## ✨ Features

- **Two Processing Modes**:
  - **Audio Translation**: Convert MP3 audiobooks/podcasts to different languages
  - **Text-to-Audiobook**: Create audiobooks from text files (TXT, PDF, EPUB, MOBI, DOCX, RTF, ODT, HTML)
- **High-Quality Speech Recognition**: Uses OpenAI's Whisper (large-v3) for accurate transcription
- **Multi-Language Support**: Translate to 30+ languages including Romanian, Spanish, French, German, and more
- **Natural Voice Synthesis**: Piper TTS with multiple voice options per language
- **Context-Aware Translation**: Claude Sonnet 4.5 for natural, context-aware translations
- **Wide Format Support**: Extract text from PDFs, EPUBs, MOBI ebooks, Word documents, and more
- **CPU & GPU Support**: Automatically detects and uses available hardware
- **Real-Time Progress**: Track processing status with live updates
- **Docker Ready**: Easy deployment with Docker Compose
- **Self-Hosted**: Your data stays on your server

## 💡 Use Cases

**Audio Translation:**
- Translate English audiobooks to your native language
- Localize podcasts and educational content for different audiences
- Convert English training materials to multiple languages

**Text-to-Audiobook:**
- Convert your ebook collection to audiobooks for commuting/exercising
- Create narrated versions of articles, blog posts, or research papers
- Generate multilingual audiobooks from public domain texts
- Turn documentation or manuals into audio guides
- Create accessible content for visually impaired users

## 🚀 Quick Start

### Prerequisites

**Hardware Requirements:**
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**:
  - Minimum: 16GB (for small files with CPU-only processing)
  - Recommended: 32GB (for larger files and better performance)
  - Optimal: 64GB+ (for parallel processing and large audiobooks)
- **GPU** (Optional but highly recommended):
  - NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
  - CUDA support enabled
  - Reduces processing time by 5-10x compared to CPU
- **Storage**:
  - At least 20GB free space for models and processed files
  - SSD recommended for better performance

**Software Requirements:**
- Docker 20.10+ and Docker Compose 2.0+
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- For development: Python 3.11+

**Tested Configurations:**
- Ubuntu 22.04 + RTX 4090 (24GB VRAM) + 64GB RAM: Excellent
- macOS Ventura + M2 Pro + 32GB RAM: Very Good (CPU only)
- Ubuntu 20.04 + No GPU + 16GB RAM: Functional but slow

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/open_narrator.git
cd open_narrator
```

2. **Configure environment variables**

```bash
cp .env.example .env
nano .env
```

Edit `.env` and add your configuration:

```bash
# Required: Your Anthropic API key
ANTHROPIC_API_KEY=your_api_key_here

# Optional: Model configurations (defaults shown)
WHISPER_MODEL=large-v3              # Options: tiny, base, small, medium, large-v3
WHISPER_COMPUTE_TYPE=auto           # Options: auto, int8, float16
TTS_ENGINE=piper                    # Currently only piper supported
MAX_UPLOAD_SIZE_MB=500              # Maximum audio file size
DEBUG=false                         # Set to true for verbose logging
```

3. **Start the application**

```bash
docker-compose up -d
```

First startup will take several minutes to download models (Whisper: ~3GB, Piper voices: varies).

4. **Access the web interface**

Open your browser and navigate to: `http://localhost:8000`

### GPU Support (Optional)

If you have an NVIDIA GPU, enable GPU acceleration for 5-10x faster processing:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Restart the container:

```bash
docker-compose down
docker-compose up -d
```

## 📖 Usage

### Mode 1: Audio Translation (MP3 → MP3)

1. **Upload Audio**: Drag and drop your MP3 file(s)
2. **Select Languages**: Choose source (e.g., English) and target language (e.g., Romanian)
3. **Choose Voice**: Select a voice with preview playback
4. **Add Context** (optional): Provide context about the audio for better translation
5. **Process**: Start the translation pipeline
6. **Monitor**: Track progress in real-time (transcription → translation → speech synthesis)
7. **Download**: Get your translated audiobook

### Mode 2: Text-to-Audiobook (Document → MP3)

1. **Upload Text File**: Drag and drop your document (TXT, PDF, EPUB, MOBI, DOCX, RTF, ODT, HTML)
2. **Select Language**: Choose target language for narration (or keep original)
3. **Choose Voice**: Select a voice with preview playback
4. **Add Context** (optional): Provide context about the content for better translation
5. **Process**: Start the audiobook creation
6. **Monitor**: Track progress in real-time (text extraction → translation → speech synthesis)
7. **Download**: Get your professionally narrated audiobook

### Supported File Formats

- **Audio**: MP3
- **Text Documents**: TXT, MD (Markdown)
- **Ebooks**: PDF, EPUB, MOBI
- **Office Documents**: DOCX (Word), RTF, ODT (OpenDocument)
- **Web**: HTML, HTM

## 🏗️ Architecture

### Audio Translation Pipeline
```
Upload MP3 → Transcribe (Whisper) → Translate (Claude) → Synthesize (Piper) → Download MP3
```

### Text-to-Audiobook Pipeline
```
Upload Document → Extract Text → Translate (Claude) → Synthesize (Piper) → Download MP3
```

### Pipeline Components

1. **Text Extraction** (for documents): Intelligent extraction from PDF, EPUB, MOBI, DOCX, RTF, ODT, HTML
2. **Speech-to-Text** (for audio): Whisper large-v3 transcribes audio to text with high accuracy
3. **Translation**: Claude Sonnet 4.5 translates text with context awareness and natural language understanding
4. **Text-to-Speech**: Piper TTS generates natural-sounding audio in 30+ languages with multiple voice options

## ⚙️ Configuration

### Environment Variables

See `.env.example` for all available configuration options:

- `ANTHROPIC_API_KEY`: Your Claude API key (required)
- `WHISPER_MODEL`: Whisper model size (default: large-v3)
- `WHISPER_COMPUTE_TYPE`: auto, int8, float16 (default: auto)
- `TTS_ENGINE`: TTS engine to use (default: piper)
- `MAX_UPLOAD_SIZE_MB`: Maximum file size (default: 50)

### GPU Support

To enable GPU acceleration, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔧 Troubleshooting

### Common Issues

**1. "Out of memory" errors**
- Reduce Whisper model size: Set `WHISPER_MODEL=medium` or `small` in `.env`
- Close other applications to free up RAM
- For large files (>1 hour), use GPU acceleration or split the file

**2. "CUDA out of memory" (GPU)**
- Use smaller Whisper model: `WHISPER_MODEL=medium`
- Reduce batch size in processing
- Ensure no other GPU applications are running

**3. "Anthropic API rate limit exceeded"**
- The translation service implements exponential backoff and will retry automatically
- For very large files, processing may take longer due to rate limits
- Consider upgrading your Anthropic API tier for higher limits

**4. Docker container fails to start**
```bash
# Check logs
docker-compose logs -f

# Common fixes:
# - Ensure port 8000 is not already in use
# - Verify .env file exists and contains ANTHROPIC_API_KEY
# - Check Docker has enough memory allocated (8GB+ recommended)
```

**5. Models not downloading**
- Ensure you have stable internet connection
- Verify you have at least 20GB free disk space
- Check Docker volume mounts are correct

**6. Poor audio quality in output**
- Try a different voice for the target language
- For audio translation: Ensure input audio is high quality (64kbps+ MP3)
- For text-to-audiobook: Verify text extraction was successful (check job logs)
- Check that the translation is accurate (poor translation = poor narration)

**7. Slow processing (CPU-only)**
- Expected processing time: ~10-15 minutes per hour of audio on modern CPU
- Consider using GPU for 5-10x speedup
- Use smaller Whisper model: `WHISPER_MODEL=medium` or `small`

**8. Text extraction fails**
- **PDF issues**: Some PDFs are scanned images without text. Use OCR software first or try a text-based PDF
- **EPUB/MOBI**: Ensure the ebook file is not DRM-protected
- **DOCX**: Modern DOCX files work best. For older DOC files, convert to DOCX first
- **Encoding issues**: Try saving your TXT file as UTF-8 encoding
- Check Docker logs for specific error: `docker-compose logs app`

**9. Health check failing**
```bash
# Test health endpoint manually
curl http://localhost:8000/health

# If it fails, check:
# - Application logs: docker-compose logs app
# - Port forwarding: docker ps
# - Firewall settings
```

### Getting Help

- Check [GitHub Issues](https://github.com/yourusername/open_narrator/issues) for known problems
- Search [Discussions](https://github.com/yourusername/open_narrator/discussions) for solutions
- Create a new issue with:
  - Your system specs (OS, RAM, GPU)
  - Docker logs: `docker-compose logs --tail=100`
  - Steps to reproduce the problem

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Initialize database
python -c "from app.database import init_db; init_db()"

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest -v --cov=app

# Run linters
black app/ tests/ scripts/
ruff check app/ tests/ scripts/ --fix
mypy app/ scripts/
```

### Project Structure

```
open_narrator/
├── app/                    # Application code
│   ├── api/               # API routes
│   ├── services/          # Business logic
│   │   ├── stt_service.py          # Speech-to-text (Whisper)
│   │   ├── text_extraction_service.py  # Document text extraction
│   │   ├── translation_service.py  # Claude translation
│   │   ├── tts_service.py          # Text-to-speech (Piper)
│   │   └── pipeline.py             # Processing pipeline
│   ├── providers/         # LLM provider abstractions
│   ├── tts_engines/       # TTS engine abstractions
│   └── templates/         # HTML templates
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── data/                  # Runtime data
└── docker/                # Docker configuration
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests and linters: `pytest && black . && ruff check . && mypy app/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Quality Standards

- All code must pass `black`, `ruff`, and `mypy` checks
- Maintain test coverage above 80%
- Add type hints to all functions
- Write comprehensive docstrings (Google style)
- Add tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Anthropic Claude](https://www.anthropic.com/) for natural language translation
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech synthesis
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyPDF2](https://github.com/py-pdf/pypdf2) for PDF text extraction
- [ebooklib](https://github.com/aerkalov/ebooklib) for EPUB support
- [python-docx](https://github.com/python-openxml/python-docx) for Word document support
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing

## 📧 Support

For issues and questions, please use [GitHub Issues](https://github.com/yourusername/open_narrator/issues).

## 🗺️ Roadmap

- [x] ✅ Text-to-audiobook creation from documents (PDF, EPUB, MOBI, DOCX, etc.)
- [ ] OCR support for scanned PDFs and images
- [ ] Support for additional TTS engines (XTTS-v2, F5-TTS, ElevenLabs)
- [ ] Batch processing improvements (multiple files at once)
- [ ] Voice cloning support for personalized narration
- [ ] Multi-user authentication and user management
- [ ] Cloud storage integration (S3, Google Drive, Dropbox)
- [ ] Advanced audio preprocessing (noise reduction, normalization)
- [ ] Custom translation prompts and fine-tuning
- [ ] Support for more LLM providers (OpenAI, Google Gemini, local models)
- [ ] Chapter detection and splitting for long audiobooks
- [ ] Subtitle/caption generation (SRT, VTT)
