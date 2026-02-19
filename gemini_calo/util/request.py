"""
HTTP request and response utilities for gemini-calo.

Provides centralized handling for compression, decompression, and HTTP client
configuration to eliminate code duplication and ensure consistency.
"""
import gzip
import zlib
from typing import Dict, Optional

import httpx


def decompress_content(content: bytes, content_encoding: Optional[str]) -> bytes:
    """
    Decompress content if it's gzip or deflate encoded.
    
    Args:
        content: Raw response content
        content_encoding: Content-Encoding header value
        
    Returns:
        Decompressed content or original content if not compressed
        
    Raises:
        gzip.BadGzipFile: If gzip decompression fails (caller should handle)
        zlib.error: If deflate decompression fails (caller should handle)
    """
    if not content_encoding:
        return content
    encoding = normalize_encoding(content_encoding)
    if 'gzip' in encoding or 'x-gzip' in encoding:
        try:
            return gzip.decompress(content)
        except (gzip.BadGzipFile, EOFError):
            # If decompression fails, return original content
            return content
    elif 'deflate' in encoding:
        try:
            # Try with -zlib.MAX_WBITS for raw deflate
            return zlib.decompress(content, -zlib.MAX_WBITS)
        except zlib.error:
            try:
                # Try with zlib header
                return zlib.decompress(content)
            except zlib.error:
                return content
    return content


def strip_compression_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Strip compression-related headers from response headers.
    
    Args:
        headers: Original response headers
        
    Returns:
        Headers with compression headers removed
    """
    compression_headers = {'content-encoding', 'content-length', 'transfer-encoding'}
    return {
        k: v for k, v in headers.items() 
        if k.lower() not in compression_headers
    }


def create_http_client(
    base_url: str = "https://generativelanguage.googleapis.com",
    accept_compression: bool = True,
    follow_redirects: bool = False,
    timeout: float = 300.0,
) -> httpx.AsyncClient:
    """
    Create a configured HTTP client for making requests.
    
    Args:
        base_url: Base URL for the client
        accept_compression: Whether to accept compressed responses
        follow_redirects: Whether to follow redirects
        timeout: Request timeout in seconds
        
    Returns:
        Configured httpx.AsyncClient
    """
    headers = {}
    if accept_compression:
        headers['Accept-Encoding'] = 'gzip, deflate'
    return httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        follow_redirects=follow_redirects,
        timeout=timeout,
    )


def is_compressed(content_encoding: Optional[str]) -> bool:
    """
    Check if content is compressed based on Content-Encoding header.
    
    Args:
        content_encoding: Content-Encoding header value
        
    Returns:
        True if content is compressed (gzip or deflate), False otherwise
    """
    if not content_encoding:
        return False
    encoding = normalize_encoding(content_encoding)
    return any(comp in encoding for comp in ['gzip', 'x-gzip', 'deflate'])


def normalize_encoding(encoding: str) -> str:
    """
    Normalize encoding string for consistent comparison.
    
    Args:
        encoding: Encoding string (e.g., "gzip", "GZIP", "x-gzip")
        
    Returns:
        Lowercase encoding string
    """
    return encoding.lower().strip()