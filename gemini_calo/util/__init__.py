"""
Utility functions for HTTP requests, compression, and response handling.
"""

from gemini_calo.util.request import (
    decompress_content,
    strip_compression_headers,
    create_http_client,
    is_compressed,
    normalize_encoding,
)

__all__ = [
    "decompress_content",
    "strip_compression_headers", 
    "create_http_client",
    "is_compressed",
    "normalize_encoding",
]