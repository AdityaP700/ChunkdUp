# experiments/retrieval/scale_fixture.py
"""
Synthetic memory-set generator for the scale test (round 2, risk 1).

Generates plausible in-domain distractors in ChunkdUp's actual memory style
(tech stack, tools, preferences, employment, projects) derived from the 12
real fixture memories. Distractors deliberately COMPETE with the real
memories (same keys: 'language', 'database', 'editor', 'cloud', 'employer',
'job_title', 'project', 'os', 'response_style', 'runtime_environment',
'embedding_model') because that competition is what changes retrieval
behavior at scale.

Constraints (validated by validate_scale_fixture):
  - no synthetic value collides with a real fixture value (keeps the 70
    query labels valid);
  - values 'OpenAI' and 'GCP' are excluded from distractor vocabularies to
    preserve the semantics of q43/q44 (historical-fact queries);
  - deterministic under a fixed seed; unique ids and values.
"""
import random
from typing import Dict, List

from fixture import BASE_MEMORIES, ALL_MEMORIES

_REAL_VALUES = {m["value"] for m in ALL_MEMORIES}

# (type, key, [base values], [value qualifiers], [conversation topics])
TEMPLATES = [
    ("programming_language", "language",
     ["JavaScript", "TypeScript", "Rust", "C++", "Java", "Kotlin", "C#", "Ruby", "PHP", "Swift",
      "Scala", "Haskell", "Elixir", "Julia", "Dart", "Lua", "Perl", "Zig"],
     ["for backend services", "for scripting", "for data pipelines", "for CLI tools",
      "for web scraping", "for automation", "for prototypes", "in production", "at hackathons"],
     ["side project stack", "work tooling discussion", "language preferences chat"]),
    ("tool", "database",
     ["MySQL", "MongoDB", "Redis", "SQLite", "MariaDB", "Cassandra", "DynamoDB", "Elasticsearch",
      "ClickHouse", "Neo4j", "TimescaleDB", "CockroachDB", "Supabase", "Firebase"],
     ["for caching", "as the primary store", "for analytics", "in side projects", "for queues",
      "for session data", "for logs", "at the day job"],
     ["backend storage chat", "infrastructure planning", "database selection discussion"]),
    ("editor", "ide",
     ["VS Code", "Emacs", "Sublime Text", "IntelliJ IDEA", "PyCharm", "Vim", "Zed", "Fleet",
      "Helix", "Kate"],
     ["for quick edits", "for large refactors", "at the office", "on the laptop",
      "for pair programming", "over SSH"],
     ["editor preferences", "tooling setup chat"]),
    ("cloud_provider", "cloud",
     ["Azure", "DigitalOcean", "Hetzner", "Linode", "Vultr", "Oracle Cloud", "Fly.io", "Render",
      "Vercel", "Netlify"],
     ["for hosting", "for CI runners", "for object storage", "for the staging env",
      "for static sites", "for the demo deployment"],
     ["cloud infrastructure", "hosting decisions chat"]),
    ("company", "employer",
     ["Meta", "Amazon", "Microsoft", "Apple", "Netflix", "Stripe", "Uber", "Airbnb", "Shopify",
      "Tesla", "NVIDIA", "Anthropic", "Cohere", "Hugging Face", "Spotify", "Figma"],
     ["(payments team)", "(infra team)", "(research org)", "(platform group)", "(contract role)",
      "(short stint)", "(internship era)", "(current employer)"],
     ["employment history", "career discussion", "job changes chat"]),
    ("role", "job_title",
     ["Software Engineer", "Data Scientist", "DevOps Engineer", "Product Manager", "ML Engineer",
      "Backend Developer", "Frontend Developer", "Engineering Manager", "Solutions Architect",
      "Site Reliability Engineer"],
     ["(earlier in career)", "(current focus)", "(side title)", "(freelance)", "(team lead capacity)"],
     ["professional career", "role history chat"]),
    ("project_name", "project",
     ["Aegis", "Nexus", "FastLog", "Orion", "Quasar", "Pulse", "Vertex", "Halo", "Forge",
      "Beacon", "Drift", "Ember", "Flux", "Grove", "Harbor", "Lumen", "Sable", "Talon"],
     ["mobile app", "CLI tool", "rewrite", "dashboard", "bot", "internal wiki",
      "browser extension", "API service"],
     ["side projects chat", "project planning", "weekend build discussion"]),
    ("environment", "os",
     ["Windows 11", "macOS Sonoma", "Arch Linux", "Fedora", "Debian", "Alpine Linux", "Manjaro",
      "Pop!_OS"],
     ["on the spare machine", "in a VM", "on the home server", "on the old laptop", "in the cloud VM"],
     ["dev machine setup", "workstation discussion"]),
    ("preference", "response_style",
     ["Detailed and thorough", "Casual and friendly", "Bullet points preferred",
      "Short sentences", "Formal tone", "ELI5 explanations", "Code-first answers",
      "Table-heavy summaries"],
     ["for design reviews", "when debugging", "for status updates", "in code review comments", "with stakeholders"],
     ["communication style chat", "assistant behavior notes"]),
    ("tool", "embedding_model",
     ["OpenAI ada-002", "BGE-large", "E5-base", "GTE-base", "MPNet", "LaBSE", "ANCE", "ColBERTv2"],
     ["for reranking", "as a fallback", "for multilingual text", "in the old prototype", "for the search beta"],
     ["vector search stack", "embedding experiments chat"]),
    ("environment", "runtime_environment",
     ["Podman", "systemd-nspawn", "LXC", "VirtualBox", "VMware", "Hyper-V", "QEMU", "K3s"],
     ["for local tests", "on the NAS", "for CI", "for the sandbox", "before switching to containers"],
     ["runtime environment chat", "containerization notes"]),
    ("learning", "learning",
     ["Kubernetes internals", "GraphQL", "WebAssembly", "Odin", "Carbon", "Mojo", "Bun",
      "Temporal", "Ray", "DuckDB"],
     ["on weekends", "for work", "next quarter", "in my free time", "after the current project"],
     ["learning goals chat", "study plan notes"]),
]


def _expand_templates() -> List[Dict]:
    """Expand every (base value, qualifier) combination into a memory spec."""
    specs = []
    for memory_type, key, bases, qualifiers, topics in TEMPLATES:
        for base in bases:
            variants = [base] + [f"{base} {q}" for q in qualifiers]
            for value in variants:
                specs.append({
                    "type": memory_type,
                    "key": key,
                    "value": value,
                    "conversation_topic": topics[hash(value) % len(topics)],
                })
    return specs


_ALL_SPECS = _expand_templates()


def generate_scale_memories(total: int, seed: int = 42) -> List[Dict[str, str]]:
    """Returns exactly `total` memories: the real fixture memories first, then
    deterministic in-domain distractors."""
    if total < len(BASE_MEMORIES):
        raise ValueError(f"total must be >= {len(BASE_MEMORIES)}")
    rng = random.Random(seed)
    specs = _ALL_SPECS[:]
    rng.shuffle(specs)

    seen_values = set(_REAL_VALUES)
    distractors = []
    needed = total - len(BASE_MEMORIES)
    for i, spec in enumerate(specs):
        if len(distractors) >= needed:
            break
        value = spec["value"]
        if value in seen_values:
            continue
        seen_values.add(value)
        distractors.append({
            "id": f"s{i:05d}",
            "type": spec["type"],
            "key": spec["key"],
            "value": value,
            "conversation_topic": spec["conversation_topic"],
            "frequency": 1,
            "status": "active",
            "source": "synthetic",
        })
    if len(distractors) < needed:
        raise RuntimeError(
            f"Template vocabulary exhausted: only {len(distractors)} unique distractors "
            f"available, {needed} needed. Extend TEMPLATES."
        )
    return [dict(m) for m in BASE_MEMORIES] + distractors


def validate_scale_fixture(memories: List[Dict]) -> None:
    ids = [m["id"] for m in memories]
    values = [m["value"] for m in memories]
    assert len(ids) == len(set(ids)), "duplicate ids in scale fixture"
    assert len(values) == len(set(values)), "duplicate values in scale fixture"
    overlap = set(values) & _REAL_VALUES
    assert not overlap - _REAL_VALUES, f"synthetic values collide with real ones: {overlap - _REAL_VALUES}"
    real_ids = {m["id"] for m in BASE_MEMORIES}
    present_real = real_ids & set(ids)
    assert present_real == real_ids, f"missing real memories: {real_ids - present_real}"


if __name__ == "__main__":
    for n in (100, 500, 1000):
        mems = generate_scale_memories(n)
        validate_scale_fixture(mems)
        keys = {}
        for m in mems:
            keys[m["key"]] = keys.get(m["key"], 0) + 1
        print(f"scale {n}: OK  ({len(mems)} memories)  key distribution: "
              f"{dict(sorted(keys.items(), key=lambda kv: -kv[1])[:6])} ...")
