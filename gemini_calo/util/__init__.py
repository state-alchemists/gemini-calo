"""
Utility functions for HTTP requests, compression, and response handling.
"""

from gemini_calo.util.request import (
    create_http_client,
    decompress_content,
    is_compressed,
    normalize_encoding,
    strip_compression_headers,
)

__all__ = [
    "decompress_content",
    "strip_compression_headers",
    "create_http_client",
    "is_compressed",
    "normalize_encoding",
]
