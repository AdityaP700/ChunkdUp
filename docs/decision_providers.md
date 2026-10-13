# ChunkdUp Decision Providers & External Policy Integration

## Overview

ChunkdUp utilizes a pluggable `DecisionProvider` architecture to govern the memory lifecycle decisions (`STORE`, `MERGE`, `UPDATE`, `IGNORE`) whenever newly extracted facts are proposed against existing active memories.

## Laya Investigation Findings

During the architectural audit of the ChunkdUp codebase and runtime environment:
- **Package Presence**: The identifier `laya` was systematically audited across the repository, local Python environment, and standard packaging indexes. No established package or library named `laya` is installed or imported.
- **Design Decision**: Rather than fabricating artificial imports or mock modules under the name `laya`, ChunkdUp introduces a clean, provider-neutral abstraction (`DecisionProvider`) and an optional adapter (`LayaDecisionProvider` / `ExternalDecisionProvider`).
- **Safety Invariant**: When an external provider is not configured or fails (timeout, network error, malformed output), ChunkdUp gracefully and deterministically falls back to `DefaultDecisionProvider`.

## Pluggable Architecture

The decision engine supports three operational modes:

1. **`DefaultDecisionProvider` (Default)**:
   - Uses ChunkdUp's built-in `PolicyFactory` and domain-specific policies (`EnvironmentPolicy`, `PreferencePolicy`, `ToolPolicy`, `ProjectPolicy`, `EmploymentPolicy`, `RolePolicy`, `LearningPolicy`, `GeneralPolicy`).
   - Fast, deterministic, and requires no network calls.

2. **`ExternalDecisionProvider`**:
   - Supports pluggable custom Python callables or HTTP endpoints.
   - Strictly enforces response validation against the `Decision` enum.

3. **`LayaDecisionProvider`**:
   - Pre-configured adapter subclassing `ExternalDecisionProvider` that reads `LAYA_ENDPOINT` and `LAYA_API_KEY`.
   - Enabled dynamically via environment variable `CHUNKDUP_DECISION_PROVIDER=laya` or passed directly to `Memory(decision_provider=...)`.

## Wire Contract for External Providers

Any external service or HTTP webhook acting as a `DecisionProvider` must adhere to the following contract:

### Request (POST)

Headers:
```http
Content-Type: application/json
Authorization: Bearer <OPTIONAL_API_KEY>
```

Payload:
```json
{
  "existing": {
    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "key": "programming_language",
    "value": "Python",
    "type": "environment",
    "version": 1,
    "status": "active"
  },
  "proposed": {
    "key": "programming_language",
    "value": "Go",
    "type": "environment",
    "confidence": 0.95,
    "source_turn_id": "8fa85f64-5717-4562-b3fc-2c963f66afa7"
  }
}
```
*Note: If no prior active fact exists for the given key, `existing` is `null`.*

### Response (200 OK)

```json
{
  "decision": "UPDATE",
  "reason": "User explicitly adopted Go replacing Python"
}
```

Allowed values for `"decision"`:
- `"STORE"`: First-time observation of this attribute.
- `"MERGE"`: Duplicate confirmation; increments frequency without superseding.
- `"UPDATE"`: New value supersedes the prior fact (triggers append-only supersession).
- `"IGNORE"`: Discard the proposed fact.

## Timeout and Degradation

- External providers default to a strict `2.0s` timeout (`timeout_seconds`).
- If an HTTP request times out, errors with non-200, or returns invalid JSON, ChunkdUp logs a warning and falls back immediately to `DefaultDecisionProvider`.
- Agent conversations are never blocked or dropped due to remote provider unavailability.
