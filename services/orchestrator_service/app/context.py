from __future__ import annotations

from dataclasses import dataclass

from .clients.enhancer_client import EnhancerServiceClient
from .clients.rag_client import RagServiceClient
from .clients.scoring_client import ScoringServiceClient
from .clients.shaer_client import ShaerServiceClient
from .clients.yehia_client import YehiaServiceClient
from .settings import Settings


@dataclass
class ServiceRegistry:
    settings: Settings
    rag_client: RagServiceClient
    yehia_client: YehiaServiceClient
    shaer_client: ShaerServiceClient
    scoring_client: ScoringServiceClient
    enhancer_client: EnhancerServiceClient
