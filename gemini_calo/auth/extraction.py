"""Credential extraction helpers for pass-through authentication."""

from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import Request

from gemini_calo.auth.aws import AWSSigV4Auth
from gemini_calo.auth.builtin import BearerAuth, NoAuth
from gemini_calo.auth.providers import AuthProviderFunc


@dataclass
class ExtractedAWSCreds:
    """AWS credentials extracted from an incoming request.

    Attributes:
        access_key: AWS access key ID (or None if not provided)
        secret_key: AWS secret access key (or None if not provided)
        session_token: Optional session token for temporary credentials
        region: AWS region (defaults to us-east-1)
    """

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    region: str = "us-east-1"

    def is_complete(self) -> bool:
        """Check if all required credentials are present."""
        return self.access_key is not None and self.secret_key is not None


# Default header names for credential extraction
DEFAULT_AWS_ACCESS_KEY_HEADER = "X-AWS-Access-Key"
DEFAULT_AWS_SECRET_KEY_HEADER = "X-AWS-Secret-Key"
DEFAULT_AWS_SESSION_TOKEN_HEADER = "X-AWS-Session-Token"
DEFAULT_AWS_REGION_HEADER = "X-AWS-Region"
DEFAULT_AWS_BEARER_TOKEN_HEADER = "X-AWS-Bearer-Token"


def extract_aws_creds_from_headers(
    request: Request,
    access_key_header: str = DEFAULT_AWS_ACCESS_KEY_HEADER,
    secret_key_header: str = DEFAULT_AWS_SECRET_KEY_HEADER,
    session_token_header: str = DEFAULT_AWS_SESSION_TOKEN_HEADER,
    region_header: str = DEFAULT_AWS_REGION_HEADER,
    default_region: str = "us-east-1",
) -> ExtractedAWSCreds:
    """Extract AWS credentials from incoming request headers.

    This allows clients to pass their AWS credentials through the proxy
    for pass-through authentication scenarios.

    Args:
        request: The incoming FastAPI request
        access_key_header: Header name for access key (default: X-AWS-Access-Key)
        secret_key_header: Header name for secret key (default: X-AWS-Secret-Key)
        session_token_header: Header name for session token (default: X-AWS-Session-Token)
        region_header: Header name for region (default: X-AWS-Region)
        default_region: Default region if not specified in headers

    Returns:
        ExtractedAWSCreds containing the credentials found in headers.

    Example:
        >>> # Client sends:
        >>> # X-AWS-Access-Key: AKIA...
        >>> # X-AWS-Secret-Key: ...
        >>> # X-AWS-Region: ap-southeast-1
        >>> creds = extract_aws_creds_from_headers(request)
        >>> assert creds.access_key == "AKIA..."
        >>> assert creds.region == "ap-southeast-1"
    """
    return ExtractedAWSCreds(
        access_key=request.headers.get(access_key_header),
        secret_key=request.headers.get(secret_key_header),
        session_token=request.headers.get(session_token_header),
        region=request.headers.get(region_header, default_region),
    )


def create_passthrough_aws_provider(
    required: bool = True,
    default_region: str = "us-east-1",
    service: str = "bedrock",
    access_key_header: str = DEFAULT_AWS_ACCESS_KEY_HEADER,
    secret_key_header: str = DEFAULT_AWS_SECRET_KEY_HEADER,
    session_token_header: str = DEFAULT_AWS_SESSION_TOKEN_HEADER,
    region_header: str = DEFAULT_AWS_REGION_HEADER,
) -> AuthProviderFunc:
    """Factory for pass-through AWS SigV4 authentication.

    Extracts AWS credentials from incoming request headers and uses them
    to sign the upstream request. This is useful when each client has their
    own AWS credentials and you want to proxy requests without managing credentials.

    Args:
        required: If True, raises error when credentials are missing.
                  If False, falls back to NoAuth when credentials are missing.
        default_region: Default AWS region if not specified in request header.
        service: AWS service name for signing (default: bedrock).
        access_key_header: Header name for access key.
        secret_key_header: Header name for secret key.
        session_token_header: Header name for session token.
        region_header: Header name for region.

    Returns:
        An AuthProviderFunc that extracts credentials and signs requests.

    Raises:
        ValueError: If required=True and credentials are incomplete.

    Example:
        >>> provider = create_passthrough_aws_provider(
        ...     required=True,
        ...     default_region="us-east-1",
        ...     service="bedrock",
        ... )
        >>> RouteConfig(
        ...     url="https://bedrock-runtime.us-east-1.amazonaws.com",
        ...     api_keys=[],
        ...     auth=provider,
        ... )
    """

    async def provider(request: Request) -> httpx.Auth:
        creds = extract_aws_creds_from_headers(
            request,
            access_key_header=access_key_header,
            secret_key_header=secret_key_header,
            session_token_header=session_token_header,
            region_header=region_header,
            default_region=default_region,
        )

        if not creds.is_complete():
            if required:
                raise ValueError(
                    f"AWS credentials required but not provided. "
                    f"Expected headers: {access_key_header}, {secret_key_header}"
                )
            return NoAuth()

        return AWSSigV4Auth(
            access_key=creds.access_key,  # type: ignore
            secret_key=creds.secret_key,  # type: ignore
            session_token=creds.session_token,
            region=creds.region,
            service=service,
        )

    return provider


def create_passthrough_bedrock_provider(
    bearer_token_header: str = DEFAULT_AWS_BEARER_TOKEN_HEADER,
    access_key_header: str = DEFAULT_AWS_ACCESS_KEY_HEADER,
    secret_key_header: str = DEFAULT_AWS_SECRET_KEY_HEADER,
    session_token_header: str = DEFAULT_AWS_SESSION_TOKEN_HEADER,
    region_header: str = DEFAULT_AWS_REGION_HEADER,
    default_region: str = "us-east-1",
) -> AuthProviderFunc:
    """Factory for pass-through Bedrock authentication.

    Checks for a bearer token first (X-AWS-Bearer-Token → Authorization: Bearer),
    then falls back to SigV4 signing (X-AWS-Access-Key + X-AWS-Secret-Key).
    Returns NoAuth if neither is present.

    Bearer token path is used when clients hold an Amazon Bedrock API key
    (AWS_BEARER_TOKEN_BEDROCK). SigV4 path is used for standard IAM credentials.
    """

    async def provider(request: Request) -> httpx.Auth:
        bearer_token = request.headers.get(bearer_token_header)
        if bearer_token:
            return BearerAuth(token=bearer_token)

        creds = extract_aws_creds_from_headers(
            request,
            access_key_header=access_key_header,
            secret_key_header=secret_key_header,
            session_token_header=session_token_header,
            region_header=region_header,
            default_region=default_region,
        )
        if creds.is_complete():
            return AWSSigV4Auth(
                access_key=creds.access_key,  # type: ignore
                secret_key=creds.secret_key,  # type: ignore
                session_token=creds.session_token,
                region=creds.region,
                service="bedrock",
            )

        return NoAuth()

    return provider
