# ai-system/chunkdup/decision_provider.py
"""
Decision Provider interface and adapters for pluggable memory lifecycle decisions.
Supports DefaultDecisionProvider (PolicyFactory) and ExternalDecisionProvider (e.g. Laya).
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
from urllib import request as urllib_request
from urllib.error import URLError

try:
    from .policies import Decision, PolicyFactory
except ImportError:
    from policies import Decision, PolicyFactory

logger = logging.getLogger(__name__)


class DecisionProvider(ABC):
    """Abstract interface for memory lifecycle policy decision providers."""

    @abstractmethod
    def decide(
        self,
        existing_memory: Optional[Dict[str, Any]],
        proposed_memory: Dict[str, Any],
    ) -> Decision:
        """Determine lifecycle decision (STORE, MERGE, UPDATE, IGNORE) for a fact."""
        pass

    def decide_batch(
        self,
        pairs: List[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]],
    ) -> List[Decision]:
        """Batch evaluate decisions for multiple (existing, proposed) pairs."""
        return [self.decide(existing, proposed) for existing, proposed in pairs]


class DefaultDecisionProvider(DecisionProvider):
    """
    Default rule-based decision provider.
    Wraps ChunkdUp's proven PolicyFactory and BasePolicy implementations.
    """

    def decide(
        self,
        existing_memory: Optional[Dict[str, Any]],
        proposed_memory: Dict[str, Any],
    ) -> Decision:
        policy = PolicyFactory.get(proposed_memory.get("type", "general"))
        if not policy:
            return Decision.STORE
        return policy.decide(existing_memory, proposed_memory)


class ExternalDecisionProvider(DecisionProvider):
    """
    Adapter for remote decision providers (such as Laya).
    Sends typed fact proposals over HTTP or local callable, with fallback to DefaultDecisionProvider.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        callback: Optional[Callable[[Optional[Dict[str, Any]], Dict[str, Any]], Optional[str]]] = None,
        timeout_seconds: float = 2.0,
        fallback_provider: Optional[DecisionProvider] = None,
    ):
        self.endpoint_url = endpoint_url
        self.callback = callback
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback_provider or DefaultDecisionProvider()

    def decide(
        self,
        existing_memory: Optional[Dict[str, Any]],
        proposed_memory: Dict[str, Any],
    ) -> Decision:
        # 1. Custom callable callback
        if self.callback is not None:
            try:
                raw_decision = self.callback(existing_memory, proposed_memory)
                if raw_decision:
                    return self._parse_decision(raw_decision)
            except Exception as e:
                logger.warning(f"External decision callback failed: {e}. Falling back to default.")
                return self.fallback.decide(existing_memory, proposed_memory)

        # 2. Remote HTTP endpoint
        if self.endpoint_url:
            try:
                payload = json.dumps({
                    "existing": existing_memory,
                    "proposed": proposed_memory,
                }).encode("utf-8")
                req = urllib_request.Request(
                    self.endpoint_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib_request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        raw_decision = data.get("decision")
                        if raw_decision:
                            return self._parse_decision(raw_decision)
            except Exception as e:
                logger.warning(
                    f"External decision request to {self.endpoint_url} failed: {e}. "
                    "Falling back to DefaultDecisionProvider."
                )
                return self.fallback.decide(existing_memory, proposed_memory)

        # If no endpoint or callback configured, use fallback
        return self.fallback.decide(existing_memory, proposed_memory)

    @staticmethod
    def _parse_decision(value: Any) -> Decision:
        if isinstance(value, Decision):
            return value
        val_str = str(value).lower().strip()
        for d in Decision:
            if d.value == val_str:
                return d
        raise ValueError(f"Unrecognized decision value from provider: {value}")


class LayaDecisionProvider(ExternalDecisionProvider):
    """
    Dedicated provider adapter for Laya decision engine.
    Configured via LAYA_ENDPOINT and LAYA_API_KEY environment variables.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 2.0,
        fallback_provider: Optional[DecisionProvider] = None,
    ):
        url = endpoint_url or os.getenv("LAYA_ENDPOINT")
        self.api_key = api_key or os.getenv("LAYA_API_KEY")
        super().__init__(
            endpoint_url=url,
            timeout_seconds=timeout_seconds,
            fallback_provider=fallback_provider,
        )


def get_decision_provider(provider_name: Optional[str] = None) -> DecisionProvider:
    """Factory function to resolve configured DecisionProvider."""
    name = (provider_name or os.getenv("CHUNKDUP_DECISION_PROVIDER", "default")).lower().strip()
    if name == "default":
        return DefaultDecisionProvider()
    elif name in ("laya", "external"):
        return LayaDecisionProvider()
    else:
        logger.warning(f"Unknown decision provider '{name}', using default.")
        return DefaultDecisionProvider()
