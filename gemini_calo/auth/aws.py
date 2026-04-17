"""AWS SigV4 authentication provider for Bedrock and other AWS services."""

from dataclasses import dataclass
from typing import Generator, Optional

import httpx
from fastapi import Request

from gemini_calo.auth.providers import AuthProviderFunc


@dataclass
class AWSCredentials:
    """AWS credentials for SigV4 signing.

    Attributes:
        access_key: AWS access key ID
        secret_key: AWS secret access key
        session_token: Optional session token for temporary credentials
        region: AWS region (default: us-east-1)
        service: AWS service name (default: bedrock)
    """

    access_key: str
    secret_key: str
    session_token: Optional[str] = None
    region: str = "us-east-1"
    service: str = "bedrock"


@dataclass
class AWSSigV4Auth(httpx.Auth):
    """httpx.Auth implementation for AWS SigV4 signing.

    Signs requests using AWS Signature Version 4 protocol.
    Required for AWS Bedrock, API Gateway with IAM auth, and other AWS services.

    The signing process requires the full request (method, URL, headers, body)
    to compute the signature, which is why this is implemented as an httpx.Auth
    rather than simple header injection.

    Attributes:
        access_key: AWS access key ID
        secret_key: AWS secret access key
        session_token: Optional session token for temporary credentials
        region: AWS region
        service: AWS service name

    Note:
        AWS SigV4 signing requires the request body for computing the payload hash.
        For streaming requests, this implementation will buffer the content for
        signing. After calling request.read(), the content is cached internally
        and will be used when the request is sent. This may have memory implications
        for very large streaming uploads.
    """

    access_key: str
    secret_key: str
    session_token: Optional[str] = None
    region: str = "us-east-1"
    service: str = "bedrock"

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Sign the request with AWS SigV4 signature.

        This implementation uses botocore's SigV4Auth for the actual signing,
        but adapts it to work with httpx requests.

        For streaming requests, the content is buffered by calling request.read()
        and then cached internally for sending.
        """
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        # Get body content for signing
        # For streaming requests, we need to call read() to buffer the content
        # This raises RequestNotRead if content hasn't been read yet
        try:
            body_content = request.content
        except httpx.RequestNotRead:
            # Streaming request - buffer the content
            # This modifies the request in-place, caching the content
            body_content = request.read()

        # Create AWS request for signing
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=body_content if body_content else b"",
            headers=dict(request.headers),
        )

        # Ensure host header is present (required for SigV4)
        if "host" not in aws_request.headers:
            aws_request.headers["host"] = request.url.host
            if request.url.port and request.url.port not in (80, 443):
                aws_request.headers["host"] = f"{request.url.host}:{request.url.port}"

        # Add security token header if using temporary credentials
        if self.session_token:
            aws_request.headers["x-amz-security-token"] = self.session_token

        # Create credentials and sign the request
        creds = Credentials(
            access_key=self.access_key,
            secret_key=self.secret_key,
            token=self.session_token,
        )

        signer = SigV4Auth(creds, self.service, self.region)
        signer.add_auth(aws_request)

        # Apply signed headers back to httpx request
        for key in ["Authorization", "X-Amz-Date", "X-Amz-Security-Token"]:
            if key in aws_request.headers:
                request.headers[key] = aws_request.headers[key]

        # Also copy the x-amz-content-sha256 if present
        for key in aws_request.headers:
            if key.lower().startswith("x-amz"):
                if key not in request.headers:
                    request.headers[key] = aws_request.headers[key]

        yield request


def create_aws_sigv4_provider(
    credentials: AWSCredentials,
) -> AuthProviderFunc:
    """Factory for AWS SigV4 authentication with static credentials.

    Use this when you want to use the same AWS credentials for all requests
    to a route (e.g., your own AWS account credentials for Bedrock).

    Args:
        credentials: AWSCredentials containing access key, secret key, etc.

    Returns:
        An AuthProviderFunc that signs requests with the provided credentials.

    Example:
        >>> creds = AWSCredentials(
        ...     access_key="AKIA...",
        ...     secret_key="...",
        ...     region="us-east-1",
        ...     service="bedrock",
        ... )
        >>> provider = create_aws_sigv4_provider(creds)
        >>> RouteConfig(
        ...     url="https://bedrock-runtime.us-east-1.amazonaws.com",
        ...     api_keys=[],
        ...     auth=provider,
        ... )
    """

    async def provider(request: Request) -> httpx.Auth:
        return AWSSigV4Auth(
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            session_token=credentials.session_token,
            region=credentials.region,
            service=credentials.service,
        )

    return provider
