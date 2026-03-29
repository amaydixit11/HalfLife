"""
test_benchmark.py — Validates benchmark metrics and corpus construction.

These tests verify the evaluation machinery itself, not HalfLife's
ranking quality. If these pass, you can trust the benchmark numbers.
"""

import pytest
from datetime import datetime, timedelta, timezone

from scripts.corpus import build_corpus, CorpusChunk, _vintage_timestamp
from scripts.benchmark import ndcg_at_k, mrr, temporal_freshness, aggregate_results


# --------------------------------------------------------------------------- #
#  Corpus construction                                                         #
# --------------------------------------------------------------------------- #

class TestCorpus:
    def setup_method(self):
        self.chunks, self.queries = build_corpus()
        self.chunk_map = {c.chunk_id: c for c in self.chunks}

    def test_corpus_size(self):
        # 3 topics × 3 doc_types × 3 vintages × 2 = 54
        assert len(self.chunks) == 54

    def test_query_count(self):
        # 3 topics × 3 intents × 4 queries = 36
        assert len(self.queries) == 36

    def test_all_vintages_present(self):
        vintages = {c.vintage for c in self.chunks}
        assert vintages == {"recent", "mid", "old"}

    def test_all_doc_types_present(self):
        doc_types = {c.doc_type for c in self.chunks}
        assert doc_types == {"news", "research", "documentation"}

    def test_all_intents_present(self):
        intents = {q.intent for q in self.queries}
        assert intents == {"fresh", "historical", "static"}

    def test_timestamps_are_timezone_aware(self):
        for chunk in self.chunks:
            assert chunk.timestamp.tzinfo is not None

    def test_recent_chunks_are_newer_than_old(self):
        recent = [c for c in self.chunks if c.vintage == "recent"]
        old    = [c for c in self.chunks if c.vintage == "old"]
        assert min(c.timestamp for c in recent) > max(c.timestamp for c in old)

    def test_fresh_queries_have_recent_relevant_chunks(self):
        fresh_queries = [q for q in self.queries if q.intent == "fresh"]
        for q in fresh_queries:
            relevant_chunks = [self.chunk_map[cid] for cid in q.relevant_ids]
            assert all(c.vintage == "recent" for c in relevant_chunks)

    def test_historical_queries_have_old_relevant_chunks(self):
        hist_queries = [q for q in self.queries if q.intent == "historical"]
        for q in hist_queries:
            relevant_chunks = [self.chunk_map[cid] for cid in q.relevant_ids]
            assert all(c.vintage == "old" for c in relevant_chunks)

    def test_static_queries_have_all_vintages_relevant(self):
        static_queries = [q for q in self.queries if q.intent == "static"]
        for q in static_queries:
            relevant_chunks = [self.chunk_map[cid] for cid in q.relevant_ids]
            vintages = {c.vintage for c in relevant_chunks}
            assert len(vintages) == 3

    def test_relevant_ids_are_subset_of_corpus(self):
        all_chunk_ids = {c.chunk_id for c in self.chunks}
        for q in self.queries:
            assert set(q.relevant_ids).issubset(all_chunk_ids)

    def test_relevant_ids_are_nonempty(self):
        for q in self.queries:
            assert len(q.relevant_ids) > 0

    def test_topic_isolation(self):
        """Queries should only reference relevant chunks from their own topic."""
        for q in self.queries:
            for cid in q.relevant_ids:
                chunk = self.chunk_map[cid]
                assert chunk.topic == q.topic

    def test_chunk_ids_are_unique(self):
        ids = [c.chunk_id for c in self.chunks]
        assert len(ids) == len(set(ids))

    def test_vintage_timestamps_are_ordered(self):
        now = datetime.now(timezone.utc)
        recent_age = (now - _vintage_timestamp("recent", 0)).days
        mid_age    = (now - _vintage_timestamp("mid",    0)).days
        old_age    = (now - _vintage_timestamp("old",    0)).days
        assert recent_age < mid_age < old_age


# --------------------------------------------------------------------------- #
#  nDCG metric                                                                 #
# --------------------------------------------------------------------------- #

class TestNDCG:
    def test_perfect_ranking(self):
        relevant = ["a", "b", "c"]
        retrieved = ["a", "b", "c", "d", "e"]
        assert ndcg_at_k(relevant, retrieved) == pytest.approx(1.0)

    def test_no_relevant_docs_retrieved(self):
        relevant  = ["a", "b", "c"]
        retrieved = ["d", "e", "f", "g", "h"]
        assert ndcg_at_k(relevant, retrieved) == 0.0

    def test_single_relevant_at_top(self):
        relevant  = ["a"]
        retrieved = ["a", "b", "c"]
        assert ndcg_at_k(relevant, retrieved) == pytest.approx(1.0)

    def test_later_position_scores_lower(self):
        relevant = ["target"]
        score_pos1 = ndcg_at_k(relevant, ["target", "a", "b"])
        score_pos3 = ndcg_at_k(relevant, ["a", "b", "target"])
        assert score_pos1 > score_pos3

    def test_empty_relevant(self):
        assert ndcg_at_k([], ["a", "b", "c"]) == 0.0

    def test_k_cutoff_respected(self):
        relevant  = ["a"]
        retrieved = ["b"] * 10 + ["a"]
        assert ndcg_at_k(relevant, retrieved, k=10) == 0.0


# --------------------------------------------------------------------------- #
#  MRR metric                                                                  #
# --------------------------------------------------------------------------- #

class TestMRR:
    def test_first_result_relevant(self):
        assert mrr(["a"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mrr(["b"], ["a", "b", "c"]) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert mrr(["z"], ["a", "b", "c"]) == 0.0


# --------------------------------------------------------------------------- #
#  Temporal Freshness metric                                                   #
# --------------------------------------------------------------------------- #

class TestTemporalFreshness:
    def _make_chunk(self, chunk_id: str, age_days: int) -> CorpusChunk:
        now = datetime.now(timezone.utc)
        ts  = now - timedelta(days=age_days)
        return CorpusChunk(
            chunk_id=chunk_id,
            text="dummy",
            timestamp=ts,
            doc_type="research",
            topic="test",
            vintage="recent",
            source_domain="test",
        )

    def test_fresh_chunks_score_higher(self):
        chunk_map = {
            "new": self._make_chunk("new", age_days=1),
            "old": self._make_chunk("old", age_days=365),
        }
        tf_fresh = temporal_freshness(["new"], chunk_map)
        tf_old   = temporal_freshness(["old"], chunk_map)
        assert tf_fresh > tf_old

    def test_score_bounded_0_to_1(self):
        chunk_map = {"c": self._make_chunk("c", age_days=100)}
        tf = temporal_freshness(["c"], chunk_map)
        assert 0.0 < tf <= 1.0

    def test_empty_retrieved(self):
        assert temporal_freshness([], {}) == 0.0


# --------------------------------------------------------------------------- #
#  Aggregate results                                                           #
# --------------------------------------------------------------------------- #

class TestAggregateResults:
    def _make_result(self, intent, b_ndcg, hl_ndcg, b_mrr, hl_mrr, b_tf, hl_tf):
        return {
            "query_id": "q_test",
            "query_text": "test",
            "intent": intent,
            "topic": "test",
            "detected_intent": intent,
            "baseline_ndcg":    b_ndcg,
            "halflife_ndcg":    hl_ndcg,
            "baseline_mrr":     b_mrr,
            "halflife_mrr":     hl_mrr,
            "baseline_tf":      b_tf,
            "halflife_tf":      hl_tf,
            "baseline_latency_ms":  5.0,
            "halflife_latency_ms":  8.0,
            "mean_rank_shift":      2.0,
        }

    def test_ndcg_delta_computed(self):
        results = [
            self._make_result("fresh", 0.5, 0.7, 0.4, 0.6, 0.01, 0.05),
        ]
        summary = aggregate_results(results)
        assert summary["fresh"]["ndcg_delta"] == pytest.approx(0.2)

    def test_fresh_tf_delta_positive(self):
        results = [self._make_result("fresh", 0.5, 0.7, 0.4, 0.6, 0.01, 0.05)]
        summary = aggregate_results(results)
        assert summary["fresh"]["tf_delta"] > 0

    def test_historical_tf_delta_negative(self):
        results = [self._make_result("historical", 0.5, 0.7, 0.4, 0.6, 0.04, 0.01)]
        summary = aggregate_results(results)
        assert summary["historical"]["tf_delta"] < 0
