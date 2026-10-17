# Changelog

All notable changes to the ChunkdUp memory engine and research evaluation harness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- `docs/research/associative-memory-plan.md`: Comprehensive Phase 0 research plan defining hypotheses H1-H4, experimental protocol, 7 retrieval variants, metric definitions, and failure taxonomy A-J.
- `CHANGELOG.md`: Continuous activity and milestone tracking for the Evidence-Preserving Associative Memory sprint.
- `evaluation/results/locomo_baseline_dev.json`: Reproduced clean Track A (Hit@5 = 69.10%, MRR@5 = 0.5966 on answerable Qs, Cat 3 Multi-hop Hit@5 = 30.77%) and Track B baseline (100% extraction retention failure with default developer-rule extractor) on LoCoMo development split (304 questions).

### Fixed
- `evaluation/adapters/locomo_adapter.py`: Protected `data/conversation.json` from write pollution during Track B evaluation by using isolated temporary directories (`tempfile.mkdtemp()`).
