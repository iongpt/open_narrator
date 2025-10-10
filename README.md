# OpenNarrator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Self-hosted audio translation service using Whisper, Claude, and Piper TTS

OpenNarrator is an open-source application that translates audio files (MP3) from English to multiple target languages while preserving the natural narration quality. Perfect for translating audiobooks, podcasts, and educational content.

## ✨ Features

- **High-Quality Speech Recognition**: Uses OpenAI's Whisper (large-v3) for accurate transcription
- **Multi-Language Support**: Translate to 30+ languages including Romanian
- **Natural Voice Synthesis**: Piper TTS with multiple voice options per language
- **Context-Aware Translation**: Claude Sonnet 4.5 for natural, context-aware translations
- **CPU & GPU Support**: Automatically detects and uses available hardware
- **Real-Time Progress**: Track processing status with live updates
- **Docker Ready**: Easy deployment with Docker Compose
- **Self-Hosted**: Your data stays on your server

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

1. **Upload Audio**: Drag and drop one or more MP3 files
2. **Select Languages**: Choose source (English) and target language
3. **Choose Voice**: Select a voice with preview playback
4. **Add Context** (optional): Provide context about the audio for better translation
5. **Process**: Start the translation pipeline
6. **Monitor**: Track progress in real-time
7. **Download**: Get your translated audio file

## 🏗️ Architecture

```
Upload MP3 → Transcribe (Whisper) → Translate (Claude) → Synthesize (Piper) → Download
```

### Pipeline Stages

1. **Speech-to-Text**: Whisper large-v3 transcribes audio to text
2. **Translation**: Claude Sonnet 4.5 translates text with context awareness
3. **Text-to-Speech**: Piper TTS generates natural-sounding audio

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
- Ensure input audio is high quality (64kbps+ MP3)
- Check that the translation is accurate (poor translation = poor narration)

**7. Slow processing (CPU-only)**
- Expected processing time: ~10-15 minutes per hour of audio on modern CPU
- Consider using GPU for 5-10x speedup
- Use smaller Whisper model: `WHISPER_MODEL=medium` or `small`

**8. Health check failing**
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
- [Anthropic Claude](https://www.anthropic.com/) for translation
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📧 Support

For issues and questions, please use [GitHub Issues](https://github.com/yourusername/open_narrator/issues).

## 🗺️ Roadmap

- [ ] Support for additional TTS engines (XTTS-v2, F5-TTS)
- [ ] Batch processing improvements
- [ ] Voice cloning support
- [ ] Multi-user authentication
- [ ] Cloud storage integration
- [ ] Advanced audio preprocessing
- [ ] Custom translation prompts
