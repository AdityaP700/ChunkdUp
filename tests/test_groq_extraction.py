# tests/test_groq_extraction.py
import os
import json
import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from chunkdup.extraction.schema import ExtractedFact, is_turn_eligible
from chunkdup.extraction.groq_extractor import GroqFactExtractor
from chunkdup.extraction.turn_buffer import TurnBuffer
from chunkdup.turn_repository import InMemoryTurnRepository, SourceTurnLog
from chunkdup.memory import Memory


def test_extracted_fact_schema_validation():
    """Verify ExtractedFact enforces strict validation and normalization."""
    # 1. Valid fact
    fact = ExtractedFact(
        key="Programming_Language",
        value="Python 3.13",
        type="environment",
        confidence=0.95,
        entities=["Python"],
        source_turn_id="turn-123",
    )
    assert fact.key == "programming_language"  # normalized to lowercase
    assert fact.value == "Python 3.13"
    assert fact.type == "environment"
    assert fact.confidence == 0.95
    assert fact.entities == ["Python"]

    # 2. Conversion to standard ChunkdUp memory dict
    mem_dict = fact.to_memory_dict()
    assert mem_dict["key"] == "programming_language"
    assert mem_dict["value"] == "Python 3.13"
    assert mem_dict["source_turn_id"] == "turn-123"
    assert mem_dict["status"] == "active"
    assert mem_dict["valid_to"] is None

    # 3. Invalid memory type raises ValidationError
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="os",
            value="Linux",
            type="astrology_sign",  # Disallowed type
            source_turn_id="turn-123",
        )

    # 4. Invalid key with spaces or illegal characters raises ValidationError
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="invalid key with spaces",
            value="Some Value",
            type="environment",
            source_turn_id="turn-123",
        )

    # 5. Out of bounds confidence raises ValidationError
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="os",
            value="Linux",
            type="environment",
            confidence=1.5,  # Must be <= 1.0
            source_turn_id="turn-123",
        )

    # 6. Empty value raises ValidationError
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="os",
            value="",
            type="environment",
            source_turn_id="turn-123",
        )


def test_turn_eligibility_filter():
    """Verify Level 1 eligibility check filters chit-chat and questions."""
    # Chit-chat should be ineligible
    assert not is_turn_eligible({"raw_text": "hello!"})
    assert not is_turn_eligible({"raw_text": "Good morning"})
    assert not is_turn_eligible({"raw_text": "thanks"})
    assert not is_turn_eligible({"raw_text": "ok cool"})
    assert not is_turn_eligible({"raw_text": "bye"})

    # Pure questions should be ineligible
    assert not is_turn_eligible({"raw_text": "What time is the meeting?"})
    assert not is_turn_eligible({"raw_text": "How do you calculate compound interest?"})
    assert not is_turn_eligible({"raw_text": "Where is the documentation located?"})

    # Substantive user statements should be eligible
    assert is_turn_eligible({"raw_text": "I use Python and Docker for my backend"})
    assert is_turn_eligible({"raw_text": "My preferred editor is Neovim"})
    assert is_turn_eligible({"raw_text": "I work at Acme Corp as a senior backend engineer"})


def test_mocked_groq_extraction_and_batching():
    """Verify GroqFactExtractor extracts structured facts from batched turns using mock."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_json = {
        "facts": [
            {
                "turn_id": "turn-1",
                "key": "programming_language",
                "value": "Python",
                "type": "environment",
                "confidence": 0.98,
                "entities": ["Python"],
            },
            {
                "turn_id": "turn-2",
                "key": "editor",
                "value": "Neovim",
                "type": "tool",
                "confidence": 0.95,
                "entities": ["Neovim"],
            },
        ]
    }
    mock_message.content = json.dumps(mock_json)
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    extractor = GroqFactExtractor(client=mock_client)

    turns = [
        {"turn_id": "turn-1", "speaker": "user", "raw_text": "I use Python for data pipelines"},
        {"turn_id": "turn-2", "speaker": "user", "raw_text": "My editor is Neovim"},
    ]

    facts = extractor.extract_batch(turns)
    assert len(facts) == 2

    assert facts[0].key == "programming_language"
    assert facts[0].value == "Python"
    assert facts[0].source_turn_id == "turn-1"

    assert facts[1].key == "editor"
    assert facts[1].value == "Neovim"
    assert facts[1].source_turn_id == "turn-2"


def test_mocked_groq_retry_on_429():
    """Verify GroqFactExtractor retries with exponential backoff on 429 rate limit."""
    mock_client = MagicMock()

    mock_success = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "facts": [
            {
                "turn_id": "turn-429",
                "key": "cloud_provider",
                "value": "AWS",
                "type": "environment",
                "confidence": 0.95,
                "entities": ["AWS"],
            }
        ]
    })
    mock_choice.message = mock_message
    mock_success.choices = [mock_choice]

    # First call raises 429, second call succeeds
    mock_client.chat.completions.create.side_effect = [
        Exception("Error code: 429 - Rate limit reached"),
        mock_success,
    ]

    extractor = GroqFactExtractor(client=mock_client, max_retries=3, backoff_factor=0.01)

    turn = {"turn_id": "turn-429", "speaker": "user", "raw_text": "I now deploy on AWS"}
    facts = extractor.extract_turn(turn)

    assert len(facts) == 1
    assert facts[0].key == "cloud_provider"
    assert facts[0].value == "AWS"
    assert mock_client.chat.completions.create.call_count == 2


def test_graceful_fallback_when_groq_unavailable(monkeypatch):
    """Verify extractor falls back to regex/spaCy when Groq API key is absent or call fails."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # No client, no API key
    extractor = GroqFactExtractor(api_key=None, client=None)
    assert not extractor.is_available()

    turn = {"turn_id": "turn-fb", "speaker": "user", "raw_text": "I prefer Neovim"}
    facts = extractor.extract_turn(turn)

    # Fallback to pattern rule
    assert len(facts) >= 1
    assert any(f.key == "editor" and f.value == "Neovim" for f in facts)
    assert facts[0].source_turn_id == "turn-fb"


def test_turn_buffer_layered_ingestion():
    """Verify TurnBuffer ingests raw turns instantly and batches fact extraction."""
    memory = Memory(store="memory")
    turn_log = memory.turn_log
    memory_manager = memory.manager

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "facts": [
            {
                "turn_id": "turn-b1",
                "key": "os",
                "value": "Linux",
                "type": "environment",
                "confidence": 0.95,
                "entities": ["Linux"],
            }
        ]
    })
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    extractor = GroqFactExtractor(client=mock_client)
    buffer = TurnBuffer(
        turn_log=turn_log,
        extractor=extractor,
        memory_manager=memory_manager,
        batch_size=2,
    )

    # Add turn 1 (buffer holds 1 turn, does not flush yet)
    res1 = buffer.add_turn("I use Linux", turn_id="turn-b1")
    assert res1["buffered_count"] == 1
    assert len(res1["processed_memories"]) == 0
    # Turn is already persisted in turn log (<5ms)
    assert turn_log.get("turn-b1") is not None

    # Add turn 2 (hits batch_size=2 -> flushes automatically)
    res2 = buffer.add_turn("Hello world", turn_id="turn-b2")
    assert res2["buffered_count"] == 0
    assert len(res2["processed_memories"]) == 1
    assert res2["processed_memories"][0]["key"] == "os"
    assert res2["processed_memories"][0]["value"] == "Linux"

    # Memory repository has the extracted fact
    fact = memory.repository.get_by_key("os")
    assert fact is not None
    assert fact["value"] == "Linux"
    assert fact["source_turn_id"] == "turn-b1"


def test_groq_malformed_json_fallback():
    """Verify malformed JSON from Groq triggers graceful fallback to pattern extractor."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Malformed JSON: {facts: incomplete"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    extractor = GroqFactExtractor(client=mock_client)
    turn = {"turn_id": "turn-malformed", "speaker": "user", "raw_text": "I prefer Neovim"}
    facts = extractor.extract_turn(turn)

    # Should fall back and still extract fact via regex/rules
    assert len(facts) >= 1
    assert any(f.key == "editor" and f.value == "Neovim" for f in facts)
    assert facts[0].source_turn_id == "turn-malformed"


def test_groq_api_exception_fallback():
    """Verify network or API exceptions in Groq call trigger graceful fallback."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Connection dropped")

    extractor = GroqFactExtractor(client=mock_client, max_retries=1)
    turn = {"turn_id": "turn-exc", "speaker": "user", "raw_text": "I use Linux"}
    facts = extractor.extract_turn(turn)

    # Should fall back to rule extraction
    assert len(facts) >= 1
    assert any(f.key == "os" and f.value == "Linux" for f in facts)
    assert facts[0].source_turn_id == "turn-exc"


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_live_groq_extraction():
    """Live integration test against Groq endpoint (runs only if GROQ_API_KEY is configured)."""
    extractor = GroqFactExtractor()
    assert extractor.is_available()

    turn = {
        "turn_id": "live-test-1",
        "speaker": "user",
        "raw_text": "I am a staff software engineer and I primarily use Rust",
    }
    facts = extractor.extract_turn(turn)
    assert len(facts) >= 1
    keys = {f.key for f in facts}
    assert "programming_language" in keys or "role" in keys
