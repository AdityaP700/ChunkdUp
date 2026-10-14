"""
Phase 2 Diagnostic: Measure Graph Coverage, Provenance, Reachability, and Evidence Presence.
Analyzes why pure PPR has low recall on LoCoMo conversations.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

import json
from evaluation.adapters.locomo_adapter import LoCoMoLoader
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph
from chunkdup.retrieval_ppr import RelationalPPRGraph

locomo_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
dev_samples, _ = LoCoMoLoader.load(locomo_path, split="dev")

diagnostic_report = {}

for sample in dev_samples:
    print(f"\n=======================================================")
    print(f"Graph Diagnostics for {sample.sample_id} ({len(sample.turns)} turns, {len(sample.questions)} questions)")
    print(f"=======================================================")

    rel_extractor = RelationalExtractor(mode="rules")
    rel_graph = RelationalGraph()

    turn_dicts = [
        {"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time}
        for t in sample.turns
    ]

    claims_extracted = []
    for idx, t in enumerate(turn_dicts):
        ctx_window = turn_dicts[max(0, idx - 3):idx]
        claims = rel_extractor.extract_from_window(t, context_window=ctx_window)
        for c in claims:
            rel_graph.add_claim(c)
            claims_extracted.append(c)

    passages = [{"id": t.dia_id, "text": f"{t.speaker}: {t.text}"} for t in sample.turns]
    ppr_graph = RelationalPPRGraph(relational_graph=rel_graph)
    ppr_graph.build_from_passages_and_claims(passages)

    # 1. Number of entities, claims, passages in graph
    passage_nodes = [node for node, ntype in zip(ppr_graph.idx_to_node, ppr_graph.node_types) if ntype == "passage"]
    claim_nodes = [node for node, ntype in zip(ppr_graph.idx_to_node, ppr_graph.node_types) if ntype == "claim"]
    entity_nodes = [node for node, ntype in zip(ppr_graph.idx_to_node, ppr_graph.node_types) if ntype == "entity"]
    num_edges = sum(len(v) for v in ppr_graph.adj.values()) // 2

    # 2. Percentage of gold evidence turns represented in graph
    all_passage_ids = {node.split("passage:", 1)[1] for node in passage_nodes if node.startswith("passage:")}
    all_gold_turns = set()
    for q in sample.questions:
        all_gold_turns.update(q.evidence_dia_ids)

    gold_in_passages = all_gold_turns.intersection(all_passage_ids)
    pct_gold_in_passages = (len(gold_in_passages) / len(all_gold_turns)) if all_gold_turns else 1.0

    # 3. Source-turn provenance on claims
    claims_with_turn = [c for c in claims_extracted if c.source_turn_ids]
    pct_claims_with_turn = len(claims_with_turn) / len(claims_extracted) if claims_extracted else 0.0

    # Check how many gold turns have associated claims
    turns_with_claims = set()
    for c in claims_extracted:
        turns_with_claims.update(c.source_turn_ids)
    gold_turns_with_claims = all_gold_turns.intersection(turns_with_claims)
    pct_gold_with_claims = (len(gold_turns_with_claims) / len(all_gold_turns)) if all_gold_turns else 0.0

    # 4. Reachability from query seeds
    total_answerable = 0
    ppr_top5_hits = 0
    seeds_extracted_count = 0
    seeds_matched_gold_count = 0

    for q in sample.questions:
        if not q.is_answerable or not q.evidence_dia_ids:
            continue
        total_answerable += 1
        gt_turns = set(q.evidence_dia_ids)

        # Entities extracted for query
        entities = ppr_graph._extract_query_entities(q.question)
        if entities:
            seeds_extracted_count += 1
            for ent in entities:
                ent_key = f"ent:{ent}"
                if ent_key in ppr_graph.node_to_idx:
                    # Check if this entity is connected to any gold turn
                    ent_idx = ppr_graph.node_to_idx[ent_key]
                    nbrs = ppr_graph.adj.get(ent_idx, {})
                    nbr_keys = {ppr_graph.idx_to_node[n_idx] for n_idx in nbrs}
                    gold_passage_keys = {f"passage:{tid}" for tid in gt_turns}
                    if nbr_keys & gold_passage_keys:
                        seeds_matched_gold_count += 1
                        break

        res = ppr_graph.retrieve(query=q.question, top_k=5)
        ret_tids = {r.get("id") for r in res}
        if ret_tids & gt_turns:
            ppr_top5_hits += 1

    sample_diag = {
        "sample_id": sample.sample_id,
        "num_passages": len(passage_nodes),
        "num_claims": len(claim_nodes),
        "num_entities": len(entity_nodes),
        "num_edges": int(num_edges),
        "total_gold_turns": len(all_gold_turns),
        "gold_in_passages": len(gold_in_passages),
        "pct_gold_in_passages": round(pct_gold_in_passages, 4),
        "total_claims": len(claims_extracted),
        "pct_claims_with_turn_provenance": round(pct_claims_with_turn, 4),
        "gold_turns_with_claims": len(gold_turns_with_claims),
        "pct_gold_with_claims": round(pct_gold_with_claims, 4),
        "total_answerable_questions": total_answerable,
        "questions_with_extracted_seeds": seeds_extracted_count,
        "seeds_matching_gold_turns": seeds_matched_gold_count,
        "ppr_top5_hits": ppr_top5_hits,
        "ppr_hit_rate": round(ppr_top5_hits / total_answerable, 4) if total_answerable else 0.0,
    }

    diagnostic_report[sample.sample_id] = sample_diag

    print(f"Passage Nodes: {len(passage_nodes)} | Claim Nodes: {len(claim_nodes)} | Entity Nodes: {len(entity_nodes)} | Edges: {num_edges}")
    print(f"Gold Evidence Turns in Graph Passages: {len(gold_in_passages)} / {len(all_gold_turns)} ({pct_gold_in_passages:.1%})")
    print(f"Claims with Source Provenance: {len(claims_with_turn)} / {len(claims_extracted)} ({pct_claims_with_turn:.1%})")
    print(f"Gold Evidence Turns Linked to Claims: {len(gold_turns_with_claims)} / {len(all_gold_turns)} ({pct_gold_with_claims:.1%})")
    print(f"Questions where Query Seeds matched Gold Turn: {seeds_matched_gold_count} / {total_answerable} ({seeds_matched_gold_count/total_answerable:.1%})")
    print(f"PPR Top-5 Evidence Hits on Answerable: {ppr_top5_hits} / {total_answerable} ({ppr_top5_hits/total_answerable:.1%})")

out_file = REPO_ROOT / "evaluation" / "results" / "graph_coverage_diagnostics.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(diagnostic_report, f, indent=2)
print(f"\nSaved graph coverage report to: {out_file}")
