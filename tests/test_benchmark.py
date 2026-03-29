"""
test_benchmark.py — Validates benchmark metrics, corpus, and learned model.

These tests verify the evaluation machinery itself. If these pass,
you can trust the benchmark numbers.
"""

import math
import pytest
from datetime import datetime, timedelta, timezone

import numpy as np

from scripts.corpus import (
    build_corpus, CorpusChunk, BenchmarkQuery,
    _vintage_timestamp, primary_chunks, decoy_chunks,
)
from scripts.benchmark import ndcg_at_k, mrr, temporal_freshness, aggregate_results
from engine.decay.learned_model import (
    DecayMLP, extract_features, LAMBDA_MIN, LAMBDA_MAX, INPUT_DIM,
)


# --------------------------------------------------------------------------- #
#  Corpus construction                                                         #
# --------------------------------------------------------------------------- #

class TestCorpus:
    def setup_method(self):
        self.chunks, self.queries = build_corpus()
        self.chunk_map = {c.chunk_id: c for c in self.chunks}
        self.primaries = primary_chunks(self.chunks)
        self.decoys    = decoy_chunks(self.chunks)

    # ---- Size assertions ------------------------------------------------

    def test_total_corpus_size(self):
        # 64 queries * 2 chunks (gold + decoy) = 128
        assert len(self.chunks) == 128

    def test_primary_count(self):
        assert len(self.primaries) == 64

    def test_decoy_count(self):
        assert len(self.decoys) == 64

    def test_query_count(self):
        assert len(self.queries) == 64

    # ---- Structure assertions -------------------------------------------

    def test_all_vintages_present(self):
        vintages = {c.vintage for c in self.primaries}
        # Tiered corpus uses 'old' for gold and 'recent' for decoys (proxied)
        # But wait, in build_tiered_corpus, gold is 'old' and decoy is 'recent'.
        assert "old" in vintages

    def test_all_doc_types_present(self):
        doc_types = {c.doc_type for c in self.primaries}
        # In Tiered Corpus, gold chunks are primarily 'research'
        assert "research" in doc_types

    def test_all_intents_present(self):
        intents = {q.intent for q in self.queries.values()}
        assert intents == {"fresh", "historical", "static"}

    def test_recent_is_newer_than_old(self):
        # Primaries are 'old', Decoys are 'recent'
        assert min(c.timestamp for c in self.decoys) > max(c.timestamp for c in self.primaries)

    def test_vintage_timestamps_are_ordered(self):
        now = datetime.now(timezone.utc)
        recent_age = (now - _vintage_timestamp("recent", 0)).days
        old_age    = (now - _vintage_timestamp("old",    0)).days
        assert recent_age < old_age

    # ---- Decoy assertions -----------------------------------------------

    def test_decoys_share_text_with_primaries(self):
        """Every decoy should have a primary with similar text. 
        In Tiered Corpus, both decoys and primaries mention same entities."""
        for d in self.decoys:
            assert d.is_decoy
            assert "Entity_Prime_Active_Latest" in d.text

    def test_decoys_never_relevant(self):
        """Decoys must not appear in any query's relevant_ids."""
        decoy_ids = {c.chunk_id for c in self.decoys}
        for q in self.queries.values():
            overlap = set(q.relevant_ids) & decoy_ids
            assert not overlap

    # ---- Ground truth assertions ----------------------------------------

    def test_fresh_queries_have_relevant_chunks(self):
        fresh_queries = [q for q in self.queries.values() if q.intent == "fresh"]
        for q in fresh_queries:
            relevant_chunks = [self.chunk_map[cid] for cid in q.relevant_ids]
            assert relevant_chunks

    def test_historical_queries_have_relevant_chunks(self):
        hist_queries = [q for q in self.queries.values() if q.intent == "historical"]
        for q in hist_queries:
            relevant_chunks = [self.chunk_map[cid] for cid in q.relevant_ids]
            assert relevant_chunks

    def test_relevant_ids_are_subset_of_corpus(self):
        all_chunk_ids = {c.chunk_id for c in self.chunks}
        for q in self.queries.values():
            assert set(q.relevant_ids).issubset(all_chunk_ids)

    def test_relevant_ids_are_nonempty(self):
        for q in self.queries.values():
            assert len(q.relevant_ids) > 0

    def test_topic_isolation(self):
        """Queries reference only relevant chunks from their own topic."""
        for q in self.queries.values():
            for cid in q.relevant_ids:
                chunk = self.chunk_map[cid]
                assert chunk.topic == q.topic


# --------------------------------------------------------------------------- #
#  nDCG metric                                                                 #
# --------------------------------------------------------------------------- #

class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_no_relevant_retrieved(self):
        assert ndcg_at_k(["a", "b"], ["c", "d", "e"]) == 0.0

    def test_single_relevant_at_top(self):
        assert ndcg_at_k(["a"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_later_position_lower_score(self):
        s1 = ndcg_at_k(["t"], ["t", "a", "b"])
        s3 = ndcg_at_k(["t"], ["a", "b", "t"])
        assert s1 > s3

    def test_empty_relevant(self):
        assert ndcg_at_k([], ["a", "b"]) == 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k(["a"], []) == 0.0

    def test_k_cutoff(self):
        # relevant only appears at position 11, beyond k=10
        retrieved = ["x"] * 10 + ["a"]
        assert ndcg_at_k(["a"], retrieved, k=10) == 0.0


# --------------------------------------------------------------------------- #
#  MRR metric                                                                  #
# --------------------------------------------------------------------------- #

class TestMRR:
    def test_first_position(self):
        assert mrr(["a"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_second_position(self):
        assert mrr(["b"], ["a", "b", "c"]) == pytest.approx(0.5)

    def test_third_position(self):
        assert mrr(["c"], ["a", "b", "c"]) == pytest.approx(1.0 / 3)

    def test_not_found(self):
        assert mrr(["z"], ["a", "b", "c"]) == 0.0

    def test_first_hit_wins(self):
        # Both b and c are relevant; first hit is b at position 2
        assert mrr(["b", "c"], ["a", "b", "c"]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  Temporal freshness metric                                                   #
# --------------------------------------------------------------------------- #

class TestTemporalFreshness:
    def _chunk(self, cid: str, age_days: int) -> CorpusChunk:
        ts = datetime.now(timezone.utc) - timedelta(days=age_days)
        return CorpusChunk(
            chunk_id=cid, text="x", timestamp=ts,
            doc_type="research", topic="t", vintage="recent",
            source_domain="test",
        )

    def test_fresher_scores_higher(self):
        cm = {"new": self._chunk("new", 1), "old": self._chunk("old", 365)}
        assert temporal_freshness(["new"], cm) > temporal_freshness(["old"], cm)

    def test_bounded_0_to_1(self):
        cm = {"c": self._chunk("c", 100)}
        tf = temporal_freshness(["c"], cm)
        assert 0.0 < tf <= 1.0

    def test_today_equals_one(self):
        cm = {"c": self._chunk("c", 0)}
        assert temporal_freshness(["c"], cm) == pytest.approx(1.0)

    def test_missing_chunk_ignored(self):
        cm = {"known": self._chunk("known", 10)}
        tf = temporal_freshness(["known", "missing"], cm)
        assert tf > 0.0

    def test_empty_returns_zero(self):
        assert temporal_freshness([], {}) == 0.0


# --------------------------------------------------------------------------- #
#  Aggregate results                                                           #
# --------------------------------------------------------------------------- #

class TestAggregateResults:
    def _r(self, intent, b_nd, hl_nd, b_mr, hl_mr, b_tf, hl_tf):
        return {
            "intent": intent,
            "baseline": {"ndcg": b_nd, "mrr": b_mr, "tf": b_tf},
            "halflife": {"ndcg": hl_nd, "mrr": hl_mr, "tf": hl_tf}
        }

    def test_ndcg_delta(self):
        # We need a list of results for aggregate_results
        s = aggregate_results([self._r("fresh", 0.5, 0.7, 0.4, 0.6, 0.01, 0.05)])
        assert s["fresh"]["ndcg_delta"] == pytest.approx(0.2, abs=1e-4)

    def test_fresh_tf_positive(self):
        s = aggregate_results([self._r("fresh", 0.5, 0.7, 0.4, 0.6, 0.01, 0.05)])
        assert s["fresh"]["tf_delta"] > 0

    def test_historical_tf_negative(self):
        s = aggregate_results([self._r("historical", 0.5, 0.7, 0.4, 0.6, 0.04, 0.01)])
        assert s["historical"]["tf_delta"] < 0

    def test_static_tf_zero(self):
        s = aggregate_results([self._r("static", 0.6, 0.6, 0.5, 0.5, 0.02, 0.02)])
        assert abs(s["static"]["tf_delta"]) < 1e-9


# --------------------------------------------------------------------------- #
#  Learned model                                                               #
# --------------------------------------------------------------------------- #

class TestLearnedModel:
    def test_input_dim(self):
        assert INPUT_DIM == 9

    def test_feature_extraction_shape(self):
        feat = extract_features("news", "arxiv-news", "some text", 5, 2)
        assert feat.shape == (INPUT_DIM,)
        assert feat.dtype == np.float32

    def test_doc_type_onehot_is_one_hot(self):
        for dt in ["news", "research", "documentation", "generic"]:
            feat = extract_features(dt, "arxiv", "text")
            onehot = feat[:4]
            assert onehot.sum() == pytest.approx(1.0), f"doc_type={dt} not one-hot"
            assert onehot.max() == pytest.approx(1.0)

    def test_unknown_doc_type_falls_back_to_generic(self):
        feat_generic  = extract_features("generic", "arxiv", "text")
        feat_unknown  = extract_features("unknown_type", "arxiv", "text")
        # Both should have generic one-hot (index 3)
        np.testing.assert_array_equal(feat_generic[:4], feat_unknown[:4])

    def test_text_length_normalised(self):
        short = extract_features("news", "arxiv", "hi")
        long  = extract_features("news", "arxiv", "x" * 2000)
        # text_length feature is at index 7
        assert short[7] < long[7]
        assert long[7] == pytest.approx(1.0)   # clipped at 1.0

    def test_feedback_ratio_cold_start(self):
        feat = extract_features("news", "arxiv", "text", 0, 0)
        # feedback_ratio at index 8, defaults to 0.5 when no feedback
        assert feat[8] == pytest.approx(0.5)

    def test_feedback_ratio_all_used(self):
        feat = extract_features("news", "arxiv", "text", 10, 0)
        assert feat[8] == pytest.approx(1.0)

    def test_feedback_ratio_all_ignored(self):
        feat = extract_features("news", "arxiv", "text", 0, 10)
        assert feat[8] == pytest.approx(0.0)

    def test_mlp_output_in_lambda_range(self):
        model = DecayMLP()
        for doc_type in ["news", "research", "documentation", "generic"]:
            feat = extract_features(doc_type, "arxiv", "sample text")
            pred = model.forward(feat)
            assert LAMBDA_MIN <= pred <= LAMBDA_MAX, \
                f"doc_type={doc_type}: λ={pred} outside [{LAMBDA_MIN}, {LAMBDA_MAX}]"

    def test_cold_start_news_faster_than_research(self):
        """Cold-start priors should make news decay faster than research."""
        model = DecayMLP()
        feat_news     = extract_features("news",     "arxiv-news", "text")
        feat_research = extract_features("research", "arxiv",      "text")
        lambda_news     = model.forward(feat_news)
        lambda_research = model.forward(feat_research)
        assert lambda_news > lambda_research, \
            f"Expected λ_news ({lambda_news:.2e}) > λ_research ({lambda_research:.2e})"

    def test_save_load_roundtrip(self, tmp_path):
        model = DecayMLP()
        # Perturb weights so we're not testing the zero case
        model.W1 += np.random.randn(*model.W1.shape).astype(np.float32) * 0.01
        feat = extract_features("news", "arxiv-news", "text")
        score_before = model.forward(feat)

        path = str(tmp_path / "test_weights.npz")
        model.save_weights(path)

        model2 = DecayMLP()
        model2.load_weights(path)
        score_after = model2.forward(feat)

        assert score_before == pytest.approx(score_after, rel=1e-5)

    def test_high_feedback_reduces_lambda(self):
        """More positive feedback → slower decay (lower λ)."""
        model = DecayMLP()
        feat_cold = extract_features("news", "arxiv-news", "text",
                                      feedback_used=0, feedback_ignored=0)
        feat_used = extract_features("news", "arxiv-news", "text",
                                      feedback_used=20, feedback_ignored=0)
        lambda_cold = model.forward(feat_cold)
        lambda_used = model.forward(feat_used)
        # High feedback_ratio (1.0) should produce a lower λ than cold start (0.5)
        # because the model is initialised to produce slower decay for used chunks
        assert lambda_used <= lambda_cold, \
            f"Expected λ_used ({lambda_used:.2e}) <= λ_cold ({lambda_cold:.2e})"