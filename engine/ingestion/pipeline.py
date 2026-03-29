import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from engine.store.redis_store import RedisStore
from engine.classifier.doc_type import DocTypeClassifier

logger = logging.getLogger(__name__)

COLLECTION_NAME = "halflife_chunks"
EMBEDDING_DIM   = 384


class HalfLifeIngestor:
    """
    Populates Qdrant and Redis from raw text chunks.

    Qdrant holds: vector, chunk_id, timestamp (ISO + epoch), doc_type,
                  source_domain, text.
    Redis holds:  decay_type, decay_params, trust_score, last_updated.

    timestamp_epoch is stored as a float payload field in Qdrant so
    range filters (e.g. "last 2 years") can use the fast indexed path
    instead of string comparison.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        redis_url:  str = "redis://localhost:6379",
    ):
        self.qdrant     = QdrantClient(url=qdrant_url)
        self.redis      = RedisStore(url=redis_url)
        self.model      = SentenceTransformer("all-MiniLM-L6-v2")
        self.classifier = DocTypeClassifier()
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    #  Collection bootstrap                                               #
    # ------------------------------------------------------------------ #

    def _ensure_collection(self) -> None:
        """
        Creates the Qdrant collection and payload indexes if they don't
        exist yet. Payload indexes are what make pre-filtering fast —
        without them Qdrant does a full scan on every filtered search.
        """
        try:
            existing = {c.name for c in self.qdrant.get_collections().collections}

            if COLLECTION_NAME not in existing:
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=rest.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=rest.Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")

            # Payload indexes — must be created explicitly after collection exists.
            # KEYWORD index on doc_type: enables equality filters like
            #   doc_type = "research"
            self.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="doc_type",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )

            # FLOAT index on timestamp_epoch: enables range filters like
            #   timestamp_epoch >= <2_years_ago_unix>
            # This is faster than filtering on the ISO string.
            self.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="timestamp_epoch",
                field_schema=rest.PayloadSchemaType.FLOAT,
            )

            # KEYWORD index on source_domain for domain-level filtering
            self.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source_domain",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )

        except Exception as e:
            # Non-fatal: indexes may already exist (Qdrant raises on duplicate)
            logger.debug(f"Collection/index setup note: {e}")

    # ------------------------------------------------------------------ #
    #  Public ingest interface                                            #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        text:          str,
        timestamp:     datetime,
        source_domain: str            = "generic",
        doc_type:      Optional[str]  = None,
        decay_type:    Optional[str]  = None,
        decay_params:  Optional[dict] = None,
        trust_score:   Optional[float] = None,
    ) -> str:
        """
        Ingest a single text chunk. Returns the assigned chunk_id.

        If doc_type / decay_type / trust_score are not provided, the
        DocTypeClassifier assigns them from the text content. This is
        the normal path — explicit overrides are for testing and
        benchmark corpus construction where you know the ground truth.

        Args:
            text:          Raw chunk text.
            timestamp:     Publication / creation datetime (tz-aware).
            source_domain: e.g. "arxiv", "github", "bbc-news", "wikipedia".
            doc_type:      Override classifier. One of: news, research,
                           documentation, reference, generic.
            decay_type:    Override decay family.
            decay_params:  Override decay parameters e.g. {"lambda": 1e-7}.
            trust_score:   Override trust prior (0.0–1.0).

        Returns:
            chunk_id (str uuid4)
        """
        # Ensure timestamp is tz-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Classifier assigns doc_type + decay priors if not overridden
        classification = self.classifier.classify(text)

        resolved_doc_type    = doc_type     or classification["doc_type"]
        resolved_decay_type  = decay_type   or classification["decay_type"]
        resolved_decay_params = decay_params or classification["decay_params"]
        resolved_trust       = trust_score  if trust_score is not None \
                               else classification["trust_score"]

        # If learned decay is explicitly requested, or if the classifier 
        # is confident, we can let the MLP predict the lambda instead of 
        # using the static priors from DocTypeClassifier.
        if resolved_decay_type == "learned":
            from engine.decay.learned_model import LEARNED_ENGINE
            predicted_lambda = LEARNED_ENGINE.predict_lambda(
                doc_type=resolved_doc_type,
                source_domain=source_domain,
                text=text
            )
            resolved_decay_params = {"lambda": predicted_lambda}


        chunk_id = str(uuid.uuid4())
        embedding = self.model.encode(text).tolist()

        # --- Qdrant write: static fields only ----------------------------
        # timestamp stored twice: ISO string for readability,
        # epoch float for fast range index queries.
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                rest.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "chunk_id":        chunk_id,
                        "text":            text,
                        "timestamp":       timestamp.isoformat(),
                        "timestamp_epoch": timestamp.timestamp(),
                        "doc_type":        resolved_doc_type,
                        "source_domain":   source_domain,
                    },
                )
            ],
        )

        # --- Redis write: mutable decay state only -----------------------
        metadata = RedisStore.build_metadata(
            chunk_id=chunk_id,
            decay_type=resolved_decay_type,
            decay_params=resolved_decay_params,
            trust_score=resolved_trust,
        )
        self.redis.set_chunk(chunk_id, metadata)

        logger.debug(
            f"Ingested chunk_id={chunk_id} doc_type={resolved_doc_type} "
            f"decay={resolved_decay_type} lambda={resolved_decay_params} "
            f"trust={resolved_trust:.2f}"
        )
        return chunk_id

    def ingest_batch(self, chunks: list[dict]) -> list[str]:
        """
        Ingest multiple chunks. Each dict must have at minimum:
            text (str), timestamp (datetime)
        Optional keys match the ingest() signature.

        Returns list of chunk_ids in the same order.
        """
        return [self.ingest(**chunk) for chunk in chunks]
