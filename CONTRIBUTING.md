# Contributing to OpenNarrator

Thank you for your interest in contributing to OpenNarrator! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guide](#code-style-guide)
- [How to Contribute](#how-to-contribute)
- [Adding New Features](#adding-new-features)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Project Architecture](#project-architecture)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/open_narrator.git
   cd open_narrator
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/open_narrator.git
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for containerized testing)
- Git

### Setup Steps

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key for testing
   ```

5. **Initialize the database:**
   ```bash
   python -c "from app.database import init_db; init_db()"
   ```

6. **Run the development server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Code Style Guide

We maintain high code quality standards. All contributions must follow these guidelines:

### Python Code Style

- **Formatting**: Use [Black](https://github.com/psf/black) for code formatting
  ```bash
  black app/ tests/ scripts/
  ```

- **Linting**: Use [Ruff](https://github.com/astral-sh/ruff) for linting
  ```bash
  ruff check app/ tests/ scripts/ --fix
  ```

- **Type Checking**: Use [MyPy](https://mypy.readthedocs.io/) for static type checking
  ```bash
  mypy app/ scripts/
  ```

### Code Quality Standards

1. **Type Hints**: All functions must have type hints
   ```python
   def process_audio(file_path: str, language: str) -> dict[str, Any]:
       """Process audio file and return results."""
       pass
   ```

2. **Docstrings**: Use Google-style docstrings for all public functions and classes
   ```python
   def translate_text(text: str, source: str, target: str) -> str:
       """Translate text from source language to target language.

       Args:
           text: The text to translate
           source: Source language code (e.g., 'en')
           target: Target language code (e.g., 'ro')

       Returns:
           Translated text

       Raises:
           ValueError: If language codes are invalid
           APIError: If translation service fails
       """
       pass
   ```

3. **Error Handling**: Always handle exceptions appropriately
   ```python
   try:
       result = api_call()
   except APIError as e:
       logger.error(f"API call failed: {e}")
       raise
   ```

4. **Logging**: Use the standard logging module
   ```python
   import logging

   logger = logging.getLogger(__name__)
   logger.info("Processing started")
   ```

5. **Imports**: Organize imports in this order:
   ```python
   # Standard library
   import os
   from typing import Any, Optional

   # Third-party
   import numpy as np
   from fastapi import FastAPI

   # Local
   from app.config import settings
   from app.models import Job
   ```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/OWNER/open_narrator/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - System information (OS, Python version, GPU/CPU)
   - Relevant logs

### Suggesting Enhancements

1. Check [Discussions](https://github.com/OWNER/open_narrator/discussions) for similar suggestions
2. Create a new discussion or issue describing:
   - The enhancement
   - Why it would be useful
   - Possible implementation approach

### Contributing Code

1. Look for issues labeled `good first issue` or `help wanted`
2. Comment on the issue to let others know you're working on it
3. Follow the development workflow below

## Adding New Features

### Adding a New LLM Provider

To add a new translation provider (e.g., OpenAI, Gemini):

1. Create a new file in `app/providers/`:
   ```python
   # app/providers/openai.py
   from app.providers.base import BaseLLMProvider

   class OpenAIProvider(BaseLLMProvider):
       async def translate(
           self,
           text: str,
           source_lang: str,
           target_lang: str,
           context: Optional[str] = None
       ) -> str:
           # Implementation
           pass
   ```

2. Register it in `app/providers/__init__.py`:
   ```python
   from app.providers.openai import OpenAIProvider

   PROVIDERS = {
       "anthropic": AnthropicProvider,
       "openai": OpenAIProvider,
   }
   ```

3. Add configuration in `app/config.py`
4. Add tests in `tests/providers/test_openai.py`
5. Update documentation

### Adding a New TTS Engine

To add a new TTS engine (e.g., XTTS, Coqui):

1. Create a new file in `app/tts_engines/`:
   ```python
   # app/tts_engines/xtts.py
   from app.tts_engines.base import BaseTTSEngine

   class XTTSEngine(BaseTTSEngine):
       def generate_audio(
           self,
           text: str,
           voice_id: str,
           language: str
       ) -> str:
           # Implementation
           pass

       def list_voices(self, language: str) -> list[Voice]:
           # Implementation
           pass
   ```

2. Register it in `app/tts_engines/__init__.py`
3. Add voice setup script in `scripts/`
4. Add tests
5. Update documentation

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chunking.py

# Run tests matching a pattern
pytest -k "test_translate"
```

### Writing Tests

1. **Unit Tests**: Test individual functions in isolation
   ```python
   def test_chunk_text():
       text = "Sample text..."
       chunks = chunk_text(text, max_tokens=1000)
       assert len(chunks) > 0
       assert all(len(c) <= 1000 for c in chunks)
   ```

2. **Integration Tests**: Test component interactions
   ```python
   async def test_translation_pipeline():
       result = await translate_and_generate(
           text="Hello world",
           target_lang="ro"
       )
       assert result.status == "completed"
   ```

3. **Mock External Services**: Use mocks for API calls
   ```python
   from unittest.mock import Mock, patch

   @patch('app.providers.anthropic.Anthropic')
   async def test_translate(mock_client):
       mock_client.messages.create.return_value = Mock(
           content=[Mock(text="Translated text")]
       )
       result = await provider.translate("Hello", "en", "ro")
       assert result == "Translated text"
   ```

### Coverage Requirements

- Minimum coverage: 80%
- Critical paths (pipeline, chunking, error handling): 100%
- New features must include tests

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**
   ```bash
   pytest -v
   ```

2. **Run all linters:**
   ```bash
   black app/ tests/ scripts/
   ruff check app/ tests/ scripts/
   mypy app/ scripts/
   ```

3. **Update documentation** if needed:
   - README.md for user-facing changes
   - Docstrings for API changes
   - CLAUDE.md for architecture changes

4. **Update CHANGELOG.md** (if exists) with your changes

### Submitting the PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub with:
   - **Title**: Clear, concise description (e.g., "Add OpenAI translation provider")
   - **Description**:
     - What changed and why
     - Related issue number (if applicable)
     - Testing performed
     - Screenshots (for UI changes)
   - **Checklist**:
     - [ ] Tests pass
     - [ ] Linters pass
     - [ ] Documentation updated
     - [ ] Type hints added
     - [ ] Follows code style guide

3. Address review feedback promptly

4. Once approved, a maintainer will merge your PR

### Commit Message Guidelines

Follow conventional commits:

- `feat: Add OpenAI translation provider`
- `fix: Handle empty audio files correctly`
- `docs: Update installation instructions`
- `test: Add tests for chunking service`
- `refactor: Simplify audio processing logic`
- `chore: Update dependencies`

## Project Architecture

### Directory Structure

```
open_narrator/
├── app/
│   ├── api/              # API routes and endpoints
│   │   ├── routes.py     # Main API routes
│   │   └── websocket.py  # SSE for progress updates
│   ├── providers/        # LLM provider abstractions
│   │   ├── base.py       # Abstract base class
│   │   └── anthropic.py  # Claude implementation
│   ├── tts_engines/      # TTS engine abstractions
│   │   ├── base.py       # Abstract base class
│   │   └── piper.py      # Piper implementation
│   ├── services/         # Business logic
│   │   ├── stt_service.py      # Speech-to-text
│   │   ├── translation_service.py  # Translation
│   │   ├── tts_service.py      # Text-to-speech
│   │   ├── chunking_service.py # Text chunking
│   │   └── pipeline.py   # End-to-end orchestration
│   ├── templates/        # HTML templates
│   ├── static/           # Static files (CSS, JS)
│   ├── models.py         # Database models
│   ├── schemas.py        # Pydantic schemas
│   ├── database.py       # Database configuration
│   ├── config.py         # Application settings
│   └── main.py           # FastAPI application
├── tests/                # Test suite
├── scripts/              # Utility scripts
└── data/                 # Runtime data
```

### Key Design Patterns

1. **Provider Pattern**: Abstract base classes for LLM providers and TTS engines
2. **Service Layer**: Business logic separated from API routes
3. **Dependency Injection**: FastAPI's dependency system for database sessions
4. **Background Tasks**: FastAPI BackgroundTasks for async processing
5. **Server-Sent Events**: Real-time progress updates

## Questions?

- Check [Discussions](https://github.com/OWNER/open_narrator/discussions)
- Ask in the issue you're working on
- Review existing code for examples

Thank you for contributing to OpenNarrator!
