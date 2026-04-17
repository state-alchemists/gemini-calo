# Auth module for Gemini Calo
# Provides extensible authentication providers for different LLM backends

from gemini_calo.auth.providers import (
    AuthConfig,
    AuthProviderFunc,
)
from gemini_calo.auth.builtin import (
    BearerAuth,
    XGoogApiKeyAuth,
    NoAuth,
    create_bearer_provider,
    create_xgoog_provider,
)
from gemini_calo.auth.aws import (
    AWSCredentials,
    AWSSigV4Auth,
    create_aws_sigv4_provider,
)
from gemini_calo.auth.extraction import (
    ExtractedAWSCreds,
    extract_aws_creds_from_headers,
    create_passthrough_aws_provider,
)

__all__ = [
    # Types
    "AuthConfig",
    "AuthProviderFunc",
    # Built-in auth implementations
    "BearerAuth",
    "XGoogApiKeyAuth",
    "NoAuth",
    "create_bearer_provider",
    "create_xgoog_provider",
    # AWS SigV4 auth
    "AWSCredentials",
    "AWSSigV4Auth",
    "create_aws_sigv4_provider",
    # Credential extraction
    "ExtractedAWSCreds",
    "extract_aws_creds_from_headers",
    "create_passthrough_aws_provider",
]