# ai-system/chunkdup/profile/profile_memory.py
"""
GPT-Inspired 3-Tier Profile, Recent Context, and Searchable Archive Memory.

Architecture:
Tier 1: Persistent Profile — Compact, bounded list of canonical facts, preferences,
        constraints, and state (max 50 facts) with 4-way consolidation (ADD, UPDATE, DELETE, NOOP).
Tier 2: Recent Context — Sliding window buffer (last 5 turns) with rolling summary.
Tier 3: Searchable Archive — Deep episodic dialogue store queried only on demand.

Hot-Path Serving:
Queries answerable by the persistent profile or recent context hit the hot path
immediately without querying the deep searchable archive.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from chunkdup.profile.schema import ProfileFact, current_iso_utc

logger = logging.getLogger(__name__)


class PersistentProfile:
    """
    Tier 1: Flat, bounded user profile storing core facts and preferences.
    """

    def __init__(self, max_facts: int = 50):
        self.max_facts = max_facts
        self.facts: Dict[str, ProfileFact] = {}  # key -> ProfileFact

    def set_fact(
        self,
        key: str,
        value: str,
        category: str = "preference",
        confidence: float = 1.0,
        source_turn_id: Optional[str] = None,
    ) -> Tuple[str, ProfileFact]:
        """
        Add or update a fact. Enforces capacity bounding via LRU/confidence eviction.
        Returns: ("ADD" or "UPDATE", ProfileFact)
        """
        clean_key = key.strip().lower()
        now = current_iso_utc()

        if clean_key in self.facts:
            existing = self.facts[clean_key]
            existing.value = value
            existing.category = category
            existing.confidence = confidence
            existing.updated_at = now
            if source_turn_id:
                existing.source_turn_id = source_turn_id
            return "UPDATE", existing

        # Check capacity bounding
        if len(self.facts) >= self.max_facts:
            self._evict_least_relevant()

        new_fact = ProfileFact(
            fact_id=str(uuid.uuid4()),
            key=clean_key,
            value=value,
            category=category,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            source_turn_id=source_turn_id,
        )
        self.facts[clean_key] = new_fact
        return "ADD", new_fact

    def delete_fact(self, key: str) -> bool:
        """
        Explicitly invalidate / delete a fact.
        """
        clean_key = key.strip().lower()
        if clean_key in self.facts:
            del self.facts[clean_key]
            return True
        return False

    def get_fact(self, key: str) -> Optional[ProfileFact]:
        return self.facts.get(key.strip().lower())

    def _evict_least_relevant(self):
        """
        Evict lowest confidence or oldest updated fact when capacity exceeded.
        """
        if not self.facts:
            return
        # Sort by confidence ascending, then updated_at ascending
        sorted_keys = sorted(
            self.facts.keys(),
            key=lambda k: (self.facts[k].confidence, self.facts[k].updated_at)
        )
        evict_key = sorted_keys[0]
        logger.info(f"Evicting profile fact '{evict_key}' due to max capacity ({self.max_facts})")
        del self.facts[evict_key]

    def search_facts(self, query: str) -> List[ProfileFact]:
        """
        Search profile facts relevant to the query.
        """
        q_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", q_lower))
        matches: List[Tuple[int, ProfileFact]] = []

        LANGUAGES = {"python", "rust", "go", "golang", "typescript", "javascript", "c++", "java", "ruby"}
        EDITORS = {"vim", "neovim", "vscode", "emacs", "pycharm", "sublime"}
        DATABASES = {"postgres", "postgresql", "mysql", "redis", "mongodb", "sqlite"}

        for fact in self.facts.values():
            key_tokens = set(re.findall(r"\w+", fact.key.lower()))
            val_tokens = set(re.findall(r"\w+", fact.value.lower()))
            cat_tokens = set(re.findall(r"\w+", fact.category.lower()))

            score = 0
            # Direct key match
            if fact.key in q_lower or any(t in q_lower for t in key_tokens if len(t) > 2):
                score += 3
            # Value match
            if fact.value.lower() in q_lower or any(t in q_lower for t in val_tokens if len(t) > 2):
                score += 2

            # Token overlap & prefix stemming (e.g. prefer <-> preferred)
            for qt in q_tokens:
                if len(qt) >= 4:
                    for kt in key_tokens.union(val_tokens):
                        if kt.startswith(qt[:4]) or qt.startswith(kt[:4]):
                            score += 1
                elif qt in key_tokens.union(val_tokens):
                    score += 1

            # Semantic domain intent
            if "language" in q_lower and ("language" in fact.key or fact.value.lower() in LANGUAGES):
                score += 4
            if "editor" in q_lower and ("editor" in fact.key or any(e in fact.value.lower() for e in EDITORS)):
                score += 4
            if "database" in q_lower and ("database" in fact.key or any(db in fact.value.lower() for db in DATABASES)):
                score += 4
            if "project" in q_lower and "project" in fact.key:
                score += 4

            if score > 0:
                matches.append((score, fact))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches]

    def to_prompt_context(self) -> str:
        """Format compact profile for LLM prompt context."""
        if not self.facts:
            return ""
        lines = ["[USER PROFILE]"]
        for fact in sorted(self.facts.values(), key=lambda f: f.key):
            lines.append(f"- {fact.key}: {fact.value}")
        return "\n".join(lines)


class RecentContextBuffer:
    """
    Tier 2: Rolling buffer of recent conversation turns and session summary.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.turns: List[Dict[str, Any]] = []
        self.rolling_summary: str = ""

    def add_turn(self, turn: Dict[str, Any]):
        self.turns.append(turn)
        if len(self.turns) > self.window_size:
            evicted = self.turns.pop(0)
            self._update_summary(evicted)

    def _update_summary(self, evicted_turn: Dict[str, Any]):
        spk = evicted_turn.get("speaker", "User")
        text = evicted_turn.get("text") or evicted_turn.get("raw_text") or ""
        snippet = f"{spk}: {text[:80]}..." if len(text) > 80 else f"{spk}: {text}"
        if not self.rolling_summary:
            self.rolling_summary = f"Prior discussion: {snippet}"
        else:
            self.rolling_summary += f" | {snippet}"

    def to_prompt_context(self) -> str:
        parts = []
        if self.rolling_summary:
            parts.append(f"[CONVERSATION SUMMARY]\n{self.rolling_summary}")
        if self.turns:
            parts.append("[RECENT TURNS]")
            for t in self.turns:
                spk = t.get("speaker", "User")
                txt = t.get("text") or t.get("raw_text") or ""
                parts.append(f"{spk}: {txt}")
        return "\n".join(parts)


class ProfileConsolidationEngine:
    """
    Decides 4-way actions: ADD, UPDATE, DELETE, NOOP on user profile facts.
    """

    def __init__(self, mode: str = "rules", groq_client: Optional[Any] = None, groq_model: Optional[str] = None):
        self.mode = mode.lower()
        self.groq_client = groq_client
        self.groq_model = groq_model or "openai/gpt-oss-20b"

    def evaluate_turn(
        self,
        turn: Dict[str, Any],
        current_profile: PersistentProfile,
    ) -> List[Dict[str, Any]]:
        """
        Analyze dialogue turn and emit list of decisions:
        [{"action": "ADD"|"UPDATE"|"DELETE"|"NOOP", "key": str, "value": str, ...}]
        """
        if self.mode == "groq" and self.groq_client is not None:
            decisions = self._evaluate_groq(turn, current_profile)
            if decisions:
                return decisions

        return self._evaluate_rules(turn, current_profile)

    def _evaluate_rules(
        self,
        turn: Dict[str, Any],
        current_profile: PersistentProfile,
    ) -> List[Dict[str, Any]]:
        text = str(turn.get("text") or turn.get("raw_text") or "").strip()
        tid = turn.get("turn_id")
        decisions: List[Dict[str, Any]] = []

        # 1. DELETE / Invalidation patterns
        del_match = re.search(
            r"(?:stop using|stopped using|don't use anymore|do not use anymore|delete my|remove my)\s+([A-Za-z0-9_\-\.\s]+)",
            text,
            re.IGNORECASE,
        )
        if del_match:
            target = del_match.group(1).strip().rstrip(".,;:!?").lower()
            # Find matching fact in current profile
            matched_key = None
            for key, fact in current_profile.facts.items():
                if target in key or target in fact.value.lower():
                    matched_key = key
                    break
            if matched_key:
                decisions.append({
                    "action": "DELETE",
                    "key": matched_key,
                    "value": "",
                    "category": "preference",
                    "source_turn_id": tid,
                    "reason": f"User explicitly stated stop using {target}",
                })

        # 2. Migration / Update patterns (e.g. moved from X to Y, switched from X to Y)
        mig_match = re.search(
            r"(?:moved from|switched from|migrated from)\s+([A-Za-z0-9_\-\.\s]+?)\s+(?:to|into)\s+([A-Za-z0-9_\-\.\s]+)",
            text,
            re.IGNORECASE,
        )
        if mig_match:
            new_val = mig_match.group(2).strip().rstrip(".,;:!?")
            # Location or tech key
            key = "location" if any(w in text.lower() for w in ("live", "moved", "city")) else "tech_stack"
            decisions.append({
                "action": "UPDATE" if key in current_profile.facts else "ADD",
                "key": key,
                "value": new_val,
                "category": "state",
                "source_turn_id": tid,
                "reason": f"User migrated to {new_val}",
            })

        # 3. Preference patterns (e.g. I prefer X, I always use X, my favorite X is Y)
        pref_match = re.search(
            r"(?:i prefer|i like to use|my favorite\s+([A-Za-z0-9_]+)\s+is)\s+([A-Za-z0-9_\-\.\s]+)",
            text,
            re.IGNORECASE,
        )
        if pref_match:
            raw_key_type = pref_match.group(1)
            raw_val = pref_match.group(2).strip().rstrip(".,;:!?")
            raw_val_lower = raw_val.lower()
            if raw_key_type:
                key = f"preferred_{raw_key_type.lower()}"
            elif raw_val_lower in ("python", "rust", "go", "golang", "typescript", "javascript", "c++", "java", "ruby"):
                key = "preferred_language"
            elif any(e in raw_val_lower for e in ("vim", "neovim", "vscode", "emacs")):
                key = "editor_preference"
            elif any(db in raw_val_lower for db in ("postgres", "postgresql", "mysql", "redis", "mongodb")):
                key = "database_preference"
            else:
                key = "preferred_tool"
            decisions.append({
                "action": "UPDATE" if key in current_profile.facts else "ADD",
                "key": key,
                "value": raw_val,
                "category": "preference",
                "source_turn_id": tid,
                "reason": f"Expressed preference for {raw_val}",
            })

        # 4. Project state patterns (e.g. I am building X, working on X)
        proj_match = re.search(
            r"(?:building|working on|developing)\s+([A-Za-z0-9_\-\.\s]+)",
            text,
            re.IGNORECASE,
        )
        if proj_match and not any(d["key"] == "current_project" for d in decisions):
            val = proj_match.group(1).strip().rstrip(".,;:!?")
            decisions.append({
                "action": "UPDATE" if "current_project" in current_profile.facts else "ADD",
                "key": "current_project",
                "value": val,
                "category": "project_state",
                "source_turn_id": tid,
                "reason": f"Active project mentioned: {val}",
            })

        # If no explicit long-term fact was identified, emit NOOP
        if not decisions:
            decisions.append({
                "action": "NOOP",
                "key": "",
                "value": "",
                "category": "ephemeral",
                "source_turn_id": tid,
                "reason": "Ephemeral conversation without persistent profile updates",
            })

        return decisions

    def _evaluate_groq(
        self,
        turn: Dict[str, Any],
        current_profile: PersistentProfile,
    ) -> List[Dict[str, Any]]:
        text = str(turn.get("text") or turn.get("raw_text") or "").strip()
        tid = turn.get("turn_id")

        system_prompt = (
            "You are ChunkdUp's Profile Memory Consolidation Engine.\n"
            "Given the user's turn and existing persistent profile facts, decide 4-way actions:\n"
            "- ADD: new persistent user fact, constraint, or preference\n"
            "- UPDATE: changed fact superseding an existing key\n"
            "- DELETE: invalidated or discontinued preference\n"
            "- NOOP: ephemeral conversation, questions, or greetings\n"
            "Return JSON:\n"
            "{\n"
            '  "decisions": [\n'
            '    {\n'
            '      "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",\n'
            '      "key": "canonical_key_name",\n'
            '      "value": "string value or empty if DELETE/NOOP",\n'
            '      "category": "preference" | "project_state" | "constraint" | "bio",\n'
            '      "reason": "short explanation"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        try:
            profile_dump = {k: f.value for k, f in current_profile.facts.items()}
            user_msg = json.dumps({"current_profile": profile_dump, "turn": {"turn_id": tid, "text": text}})
            resp = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            decisions = parsed.get("decisions", [])
            for d in decisions:
                d["source_turn_id"] = tid
            return decisions
        except Exception as e:
            logger.warning(f"Groq profile consolidation failed: {e}")
            return []


class SearchableArchive:
    """
    Tier 3: Episodic archive storing raw dialogue turns.
    """

    def __init__(self):
        self.passages: List[Dict[str, Any]] = []

    def add_turn(self, turn: Dict[str, Any]):
        self.passages.append(turn)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Lexical and token-overlap search over archive passages."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for p in self.passages:
            txt = str(p.get("text") or p.get("raw_text") or "")
            p_tokens = set(re.findall(r"\w+", txt.lower()))
            overlap = len(q_tokens.intersection(p_tokens))
            score = float(overlap) / (len(q_tokens) + 1e-5)
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": p.get("turn_id") or p.get("id"),
                "text": p.get("text") or p.get("raw_text"),
                "score": score,
                "speaker": p.get("speaker"),
            }
            for score, p in scored[:top_k]
        ]


class ProfileMemory:
    """
    Master 3-Tier GPT-Inspired Memory System:
    - Tier 1: Persistent Profile
    - Tier 2: Recent Context Window & Summary
    - Tier 3: Searchable Dialogue Archive
    """

    def __init__(
        self,
        max_profile_facts: int = 50,
        recent_window_size: int = 5,
        consolidation_mode: str = "rules",
        groq_client: Optional[Any] = None,
    ):
        self.profile = PersistentProfile(max_facts=max_profile_facts)
        self.recent_context = RecentContextBuffer(window_size=recent_window_size)
        self.archive = SearchableArchive()
        self.consolidation_engine = ProfileConsolidationEngine(
            mode=consolidation_mode,
            groq_client=groq_client,
        )
        self.decision_history: List[Dict[str, Any]] = []

    def remember(self, turn: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process incoming dialogue turn across all 3 memory tiers:
        1. Consolidate profile facts (ADD, UPDATE, DELETE, NOOP)
        2. Append to recent context buffer
        3. Index in searchable archive
        """
        decisions = self.consolidation_engine.evaluate_turn(turn, self.profile)
        applied: List[Dict[str, Any]] = []

        for d in decisions:
            action = d.get("action")
            key = d.get("key", "")
            val = d.get("value", "")
            cat = d.get("category", "preference")
            tid = d.get("source_turn_id")

            if action == "ADD" and key and val:
                _, fact = self.profile.set_fact(key, val, category=cat, source_turn_id=tid)
                applied.append({"action": "ADD", "fact": fact.to_dict()})
            elif action == "UPDATE" and key and val:
                _, fact = self.profile.set_fact(key, val, category=cat, source_turn_id=tid)
                applied.append({"action": "UPDATE", "fact": fact.to_dict()})
            elif action == "DELETE" and key:
                deleted = self.profile.delete_fact(key)
                applied.append({"action": "DELETE", "key": key, "success": deleted})
            elif action == "NOOP":
                applied.append({"action": "NOOP", "reason": d.get("reason")})

        self.decision_history.extend(applied)
        self.recent_context.add_turn(turn)
        self.archive.add_turn(turn)
        return applied

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Hot-Path Query Serving:
        - If query can be answered directly from the persistent profile:
          Serve immediately via Hot-Path WITHOUT searching deep archive.
        - Otherwise, search deep episodic archive.
        """
        profile_matches = self.profile.search_facts(query)

        # High confidence profile hit -> Hot-path serving
        if profile_matches:
            top_fact = profile_matches[0]
            return {
                "source": "profile_hot_path",
                "archive_searched": False,
                "fact": top_fact.to_dict(),
                "all_matching_facts": [f.to_dict() for f in profile_matches],
                "context_injection": self.profile.to_prompt_context(),
            }

        # Fallback to episodic archive search
        archive_results = self.archive.search(query, top_k=top_k)
        return {
            "source": "archive",
            "archive_searched": True,
            "passages": archive_results,
            "context_injection": self.recent_context.to_prompt_context(),
        }
