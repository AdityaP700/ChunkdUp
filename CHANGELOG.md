# Changelog

All notable changes to the ChunkdUp memory engine and research evaluation harness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- `docs/research/associative-memory-plan.md`: Comprehensive Phase 0 research plan defining hypotheses H1-H4, experimental protocol, 7 retrieval variants, metric definitions, and failure taxonomy A-J.
- `CHANGELOG.md`: Continuous activity and milestone tracking for the Evidence-Preserving Associative Memory sprint.
- `evaluation/results/locomo_baseline_dev.json`: Reproduced clean Track A (Hit@5 = 69.10%, MRR@5 = 0.5966 on answerable Qs, Cat 3 Multi-hop Hit@5 = 30.77%) and Track B baseline (100% extraction retention failure with default developer-rule extractor) on LoCoMo development split (304 questions).
- `ai-system/chunkdup/associative/schema.py`: Minimal evidence-preserving representations for Experience memory (raw turns), Semantic memory (atomic facts with bidirectional source turn provenance), Associative memory (typed links with vocabulary `entity_to_event`, `event_to_event`, `fact_to_source`, `state_transition`, `entity_relation`, `causal_explanation`), and optional profile attributes.
- `ai-system/chunkdup/associative/graph.py`: In-memory associative graph engine with strictly bounded traversal (`BoundedGraphExpansion`: max depth <= 2, branching factor <= 3, candidates <= 5).
- `ai-system/chunkdup/associative/retrievers.py`: All 7 experimental retrieval variants (A: `GrepRetriever`, B: `BM25Retriever`, C: `HybridCEControlRetriever`, D: `BoundedAssociativeRetriever`, E: `ConsolidatedSemanticRetriever`, F: `ConsolidatedAssociativeRetriever`, G: `QueryRoutedAssociativeRetriever`).
- `ai-system/chunkdup/associative/builder.py`: Automated representation bundle assembler constructing experience turns, semantic facts, and associative graph links from conversation samples.
- `tests/test_associative_memory.py`: Unit and integration test suite validating schema provenance, bounded expansion limits, and all 7 retrieval variants (100% pass).
- `evaluation/run_associative_sprint.py`: Turnkey unified evaluation runner for all 7 variants and controlled ablations with GPU acceleration.
- `evaluation/results/associative_sprint_dev.json` & `evaluation/results/associative_per_question_dev.csv`: Complete dev split results across 304 questions. Bounded associative expansion with branch limit $B=1$ achieves highest Hit@5 (72.53%, +3.00% over hybrid CE control) and highest MRR@5 (0.6065) while cutting latency by >50% (180ms vs 390ms). Semantic consolidation achieves lowest context tokens (144.9) while matching control accuracy.

### Fixed
- `evaluation/adapters/locomo_adapter.py`: Protected `data/conversation.json` from write pollution during Track B evaluation by using isolated temporary directories (`tempfile.mkdtemp()`).
