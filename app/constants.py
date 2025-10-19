"""Shared constants for file handling and validation."""

ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".mp4",
}

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/x-mp3",
    "audio/mpeg3",
    "audio/x-mpeg-3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac",
}

ALLOWED_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/html",
    "application/pdf",
    "application/epub+zip",
    "application/x-mobipocket-ebook",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/rtf",
    "text/rtf",
    "application/vnd.oasis.opendocument.text",
}

ALLOWED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".epub",
    ".mobi",
    ".docx",
    ".doc",
    ".rtf",
    ".odt",
    ".html",
    ".htm",
}
