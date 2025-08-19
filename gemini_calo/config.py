import logging
import os

GEMINI_API_KEYS = [
    key.strip()
    for key in os.getenv("GEMINI_CALO_API_KEYS", "").split(",")
    if key.strip() != ""
]

PROXY_API_KEYS = [
    key.strip()
    for key in os.getenv("GEMINI_CALO_PROXY_API_KEYS", "").split(",")
    if key.strip() != ""
]

HTTP_PORT = int(os.getenv("GEMINI_CALO_HTTP_PORT", "8000"))

_LOG_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARN,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}
LOG_LEVEL = _LOG_LEVEL_MAP.get(
    os.getenv("GEMINI_CALO_LOG_LEVEL", "").upper(), logging.CRITICAL
)
LOG_FILE = os.getenv("GEMINI_CALO_LOG_FILE", "app.log")