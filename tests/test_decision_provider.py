# tests/test_decision_provider.py
import pytest
from unittest.mock import MagicMock

from chunkdup.policies import Decision
from chunkdup.decision_provider import (
    DefaultDecisionProvider,
    ExternalDecisionProvider,
    LayaDecisionProvider,
    get_decision_provider,
)
from chunkdup.memory import Memory, DecisionEngine


def test_default_decision_provider_parity():
    """Verify DefaultDecisionProvider preserves standard policy behaviors."""
    provider = DefaultDecisionProvider()

    new_env = {"type": "environment", "key": "os", "value": "Linux"}
    # 1. No existing memory -> STORE
    assert provider.decide(None, new_env) == Decision.STORE

    # 2. Existing memory with same value -> MERGE
    existing_same = {"type": "environment", "key": "os", "value": "Linux"}
    assert provider.decide(existing_same, new_env) == Decision.MERGE

    # 3. Existing memory with different value -> UPDATE
    existing_diff = {"type": "environment", "key": "os", "value": "Windows"}
    assert provider.decide(existing_diff, new_env) == Decision.UPDATE


def test_external_decision_provider_callback_override():
    """Verify an external provider callback can override standard decisions."""
    # Callback that forces IGNORE for everything
    custom_cb = MagicMock(return_value="IGNORE")
    provider = ExternalDecisionProvider(callback=custom_cb)

    existing = {"type": "environment", "key": "os", "value": "Windows"}
    proposed = {"type": "environment", "key": "os", "value": "Linux"}

    decision = provider.decide(existing, proposed)
    assert decision == Decision.IGNORE
    assert custom_cb.call_count == 1


def test_external_decision_provider_fallback_on_callback_error():
    """Verify external provider falls back to DefaultDecisionProvider when callback raises."""
    def buggy_cb(existing, proposed):
        raise RuntimeError("External service connection reset")

    provider = ExternalDecisionProvider(callback=buggy_cb)

    existing = {"type": "environment", "key": "os", "value": "Windows"}
    proposed = {"type": "environment", "key": "os", "value": "Linux"}

    # Should not throw, should fall back to standard policy (UPDATE)
    decision = provider.decide(existing, proposed)
    assert decision == Decision.UPDATE


def test_external_decision_provider_fallback_on_network_failure():
    """Verify external provider falls back to default when remote endpoint fails."""
    # Point to dummy non-routable port with short timeout
    provider = ExternalDecisionProvider(
        endpoint_url="http://127.0.0.1:59999/decide",
        timeout_seconds=0.1,
    )

    existing = {"type": "tool", "key": "editor", "value": "Emacs"}
    proposed = {"type": "tool", "key": "editor", "value": "Neovim"}

    # Must cleanly fall back to UPDATE
    decision = provider.decide(existing, proposed)
    assert decision == Decision.UPDATE


def test_memory_sdk_pluggable_decision_provider():
    """Verify Memory SDK integrates custom decision provider seamlessly."""
    def always_store_cb(existing, proposed):
        return "STORE"

    custom_provider = ExternalDecisionProvider(callback=always_store_cb)
    memory = Memory(store="memory", decision_provider=custom_provider)

    # First remember
    res1 = memory.remember("I use Linux")
    assert res1["memories"][0]["decision"] == "STORE"

    # Second remember with same value would normally MERGE, but callback forces STORE
    res2 = memory.remember("I use Linux")
    assert res2["memories"][0]["decision"] == "STORE"


def test_get_decision_provider_factory():
    """Verify provider factory instantiates the correct provider."""
    p_def = get_decision_provider("default")
    assert isinstance(p_def, DefaultDecisionProvider)

    p_laya = get_decision_provider("laya")
    assert isinstance(p_laya, LayaDecisionProvider)

    p_ext = get_decision_provider("external")
    assert isinstance(p_ext, ExternalDecisionProvider)
