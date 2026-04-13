"""
Type stubs for the compiled tqdb Rust extension (tqdb.tqdb).

These stubs cover the Database class exposed via PyO3. Requires Python 3.10+.
"""

from __future__ import annotations

from typing import Any

import numpy as np

class Database:
    """Embedded vector database implementing the TurboQuant algorithm.

    Open (or create) a database with :meth:`open`. The same ``dimension``,
    ``bits``, ``seed``, and ``metric`` must be used on every reopen.
    """

    @staticmethod
    def open(
        path: str,
        dimension: int | None = None,
        bits: int = 4,
        seed: int = 42,
        metric: str = "ip",
        rerank: bool = True,
        fast_mode: bool = False,
        rerank_precision: str | None = None,
        collection: str | None = None,
        wal_flush_threshold: int | None = None,
        normalize: bool = False,
    ) -> Database:
        """Open or create a database at *path*.

        Args:
            path: Directory for database files; created if absent.
            dimension: Vector dimension. Required when creating a new database.
                Omit (or pass ``None``) to reopen an existing database — the
                dimension and other fixed parameters are read from the stored
                ``manifest.json`` automatically.
            bits: Quantization bits per coordinate (any int >= 2).
                ``2`` = highest compression (16×); ``4`` = better recall (default);
                ``8`` = near-lossless.
            seed: RNG seed for the quantizer; must match on every reopen.
            metric: Distance metric — ``"ip"`` (inner product), ``"cosine"``,
                or ``"l2"``. Fixed at creation.
            rerank: Enable dequantization-based reranking for HNSW candidates.
            fast_mode: Skip the QJL stage (~30 % faster ingest, ~5 pp recall
                loss). Default ``False``.
            rerank_precision: Reranking precision for exact reranking.
                ``None`` = dequantization reranking (no extra storage);
                ``"f16"`` = float16 (``+n×d×2`` bytes);
                ``"f32"`` = float32 (``+n×d×4`` bytes).
            collection: Subdirectory name for multi-collection setups. The
                database is opened at ``path/collection/`` when provided.
            wal_flush_threshold: Override the WAL flush threshold (number of
                entries before an automatic flush to a segment). ``None`` uses
                the compiled default.
            normalize: L2-normalise every inserted vector and every query at
                write time. Makes inner-product scoring equivalent to cosine
                similarity.
        """
        ...

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def insert(
        self,
        id: str,
        vector: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> None:
        """Insert a single vector.

        Raises:
            RuntimeError: If *id* already exists (use :meth:`upsert` to
                replace an existing entry).
        """
        ...

    def insert_batch(
        self,
        ids: list[str],
        vectors: np.ndarray | list[list[float]],
        metadatas: list[dict[str, Any] | None] | None = None,
        documents: list[str | None] | None = None,
        mode: str = "insert",
    ) -> None:
        """Batch insert (or upsert / update) vectors.

        Args:
            ids: Vector IDs.
            vectors: 2-D array of shape ``(N, D)``, float32 or float64.
            metadatas: Per-vector metadata dicts. ``None`` = no metadata.
            documents: Per-vector document strings. ``None`` = no documents.
            mode: Write mode:

                - ``"insert"`` — raises ``RuntimeError`` at the first duplicate ID.
                - ``"upsert"`` — inserts new IDs or replaces existing ones.
                - ``"update"`` — raises ``RuntimeError`` at the first missing ID.
        """
        ...

    def upsert(
        self,
        id: str,
        vector: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> None:
        """Insert or replace a single vector (always succeeds)."""
        ...

    def update(
        self,
        id: str,
        vector: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> None:
        """Update an existing vector.

        Raises:
            RuntimeError: If *id* does not exist.
        """
        ...

    def update_metadata(
        self,
        id: str,
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> None:
        """Update metadata and/or document for an existing ID without
        re-uploading the vector. The quantised representation is unchanged.

        Args:
            id: Must exist; raises ``RuntimeError`` otherwise.
            metadata: New metadata dict, or ``None`` to preserve existing.
            document: New document string, or ``None`` to preserve existing.
        """
        ...

    # ------------------------------------------------------------------ #
    # Delete operations                                                    #
    # ------------------------------------------------------------------ #

    def delete(self, id: str) -> bool:
        """Mark a vector as deleted.

        Returns:
            ``True`` if *id* existed; ``False`` if it was not found.
        """
        ...

    def delete_batch(
        self,
        ids: list[str] = [],
        where_filter: dict | None = None,
    ) -> int:
        """Delete multiple vectors.

        Args:
            ids: Explicit list of IDs to delete. Silently skips missing IDs.
                May be empty when *where_filter* is provided.
            where_filter: Optional metadata filter (same syntax as
                :meth:`search`). All matching vectors are deleted in addition
                to any IDs listed explicitly. Overlapping entries are not
                double-counted.

        Returns:
            Number of IDs that were found and deleted.
        """
        ...

    # ------------------------------------------------------------------ #
    # Read / lookup                                                        #
    # ------------------------------------------------------------------ #

    def get(self, id: str) -> dict[str, Any] | None:
        """Retrieve metadata and document for a single ID.

        Returns:
            ``{"id": str, "metadata": dict, "document": str | None}``
            or ``None`` if *id* is not found.
        """
        ...

    def get_many(self, ids: list[str]) -> list[dict[str, Any] | None]:
        """Retrieve metadata and document for multiple IDs.

        Returns:
            List aligned with *ids*; ``None`` for each ID not found.
        """
        ...

    def list_all(self) -> list[str]:
        """Return all active (non-deleted) IDs."""
        ...

    def list_ids(
        self,
        where_filter: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[str]:
        """Paginated ID list with optional metadata filter.

        Args:
            where_filter: Same filter syntax as :meth:`search`. ``None``
                returns all IDs.
            limit: Maximum number of IDs to return. ``None`` = no limit.
            offset: Number of results to skip (for pagination). Default 0.
        """
        ...

    def list_metadata_values(self, field: str) -> dict[str, int]:
        """Return a ``{value: count}`` mapping of all distinct values for
        *field* across active vectors.

        Useful for building filter UIs or faceted search. Supports dotted
        paths (e.g. ``"profile.region"``). Non-string values are stringified
        via their JSON representation.

        Returns:
            Dict mapping each unique field value to its occurrence count.
        """
        ...

    def count(self, filter: dict[str, Any] | None = None) -> int:
        """Return the number of active vectors matching an optional filter."""
        ...

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray | list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
        _use_ann: bool = False,
        ann_search_list_size: int | None = None,
        include: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for the nearest neighbours of a query vector.

        Args:
            query: 1-D query vector, float32 or float64.
            top_k: Number of results to return.
            filter: Optional metadata filter (same syntax as
                :meth:`list_ids`).
            _use_ann: Use the HNSW index when ``True`` (requires
                :meth:`create_index` to have been called). Default ``False``
                = exhaustive brute-force (highest recall).
            ann_search_list_size: HNSW ``ef_search`` override. Larger values
                improve recall at the cost of latency. Only relevant when
                ``_use_ann=True``.
            include: Subset of fields to include in each result dict.
                Valid values: ``"id"``, ``"score"``, ``"metadata"``,
                ``"document"``. Defaults to all four.

        Returns:
            List of result dicts. Each dict contains the keys requested via
            *include* (``id``, ``score``, ``metadata``, ``document``).
        """
        ...

    def query(
        self,
        query_embeddings: np.ndarray,
        n_results: int = 10,
        where_filter: dict[str, Any] | None = None,
        _use_ann: bool = False,
        ann_search_list_size: int | None = None,
        rerank_factor: int | None = None,
        include: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Search with multiple query vectors in one call.

        Args:
            query_embeddings: 2-D array of shape ``(N, D)``, float32 or
                float64.
            n_results: Results per query. Default 10.
            where_filter: Optional metadata filter.
            _use_ann: Use HNSW index when ``True``. Default ``False``.
            ann_search_list_size: HNSW ``ef_search`` override.
            rerank_factor: Oversampling multiplier for rerank candidate pool.
            include: Subset of fields to include in each result dict.
                Valid values: ``"id"``, ``"score"``, ``"metadata"``,
                ``"document"``. Defaults to all four.

        Returns:
            List of N result lists. Each result dict contains the keys
            requested via *include*.
        """
        ...

    # ------------------------------------------------------------------ #
    # Index                                                                #
    # ------------------------------------------------------------------ #

    def create_index(
        self,
        max_degree: int | None = None,
        ef_construction: int | None = None,
        search_list_size: int | None = None,
        alpha: float | None = None,
        n_refinements: int | None = None,
    ) -> None:
        """Build the HNSW graph index over all currently stored vectors.

        Call this **after** loading your data. Rebuild after large batches
        of inserts; the index is not updated incrementally.

        Vectors inserted after :meth:`create_index` are still searched
        correctly via an automatic brute-force overlay (hybrid search).

        Args:
            max_degree: Max neighbours per HNSW node. Higher = better
                recall, larger graph. Default 32.
            ef_construction: Beam width during graph construction. Higher =
                better quality, slower build. Default 200.
            search_list_size: Default ``ef_search`` for subsequent
                :meth:`search` calls with ``_use_ann=True``. Default 128.
            alpha: Edge-pruning aggressiveness. Default 1.2.
            n_refinements: Refinement passes after build. Default 5.
        """
        ...

    # ------------------------------------------------------------------ #
    # Maintenance                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return a dict of database statistics.

        Keys include:
            - ``vector_count`` — total active vectors
            - ``segment_count`` — number of immutable segment files
            - ``buffered_vectors`` — vectors in the WAL (not yet flushed)
            - ``dimension`` — vector dimension
            - ``bits`` — quantization bits
            - ``total_disk_bytes`` — total on-disk footprint
            - ``has_index`` — whether a HNSW index exists
            - ``index_nodes`` — number of indexed nodes
            - ``delta_size`` — vectors inserted after last ``create_index()`` (delta overlay)
            - ``live_codes_bytes`` — size of the in-memory codes buffer
            - ``ram_estimate_bytes`` — estimated in-memory footprint
        """
        ...

    def flush(self) -> None:
        """Flush the WAL to an immutable segment file immediately."""
        ...

    def checkpoint(self) -> None:
        """Flush pending writes and compact segments when threshold is met."""
        ...

    def close(self) -> None:
        """Flush pending data and release all file handles."""
        ...

    # ------------------------------------------------------------------ #
    # Container protocol                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Total number of active vectors (``len(db)``)."""
        ...

    def __contains__(self, id: object) -> bool:
        """``id in db`` — ``True`` if the ID exists."""
        ...


#: Alias kept for backward compatibility.
TurboQuantDB = Database
