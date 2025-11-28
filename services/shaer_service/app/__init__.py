"""
shaer_service application package.

This package exposes a FastAPI application that proxies /generate-bayt requests
to the hosted Shaer RunPod endpoint using the training-style prompt template.
"""

from .main import create_app

__all__ = ["create_app"]
