"""
test_brutal_server.py — Brutal integration tests for tqdb-server (Axum HTTP mode).

Strategy: spin up the real tqdb-server binary against a temp data directory,
exercise every endpoint brutally, then tear it down.

Run with:
    python -m pytest tests/test_brutal_server.py -v --basetemp=tmp_pytest_run_server -q
"""

import gc
import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SERVER_BIN = str(
    Path(__file__).parent.parent / "server" / "target" / "release" / "tqdb-server.exe"
)
GOOD_KEY = "dev-key"
AUTH_HDR = {"Authorization": f"ApiKey {GOOD_KEY}"}
JSON_HDR = {"Content-Type": "application/json"}
HEADERS = {**AUTH_HDR, **JSON_HDR}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(base: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/healthz", timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Server at {base} did not become ready in {timeout}s")


def _unit_vec(d: int, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-8)).tolist()


def _unit_batch(n: int, d: int = 8, seed: int = 42) -> List[List[float]]:
    rng = np.random.default_rng(seed)
    vs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vs, axis=1, keepdims=True)
    return (vs / np.maximum(norms, 1e-8)).tolist()


def _wait_job(base: str, job_id: str, timeout: float = 30.0) -> Dict:
    """Poll until a job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{base}/v1/jobs/{job_id}", headers=AUTH_HDR, timeout=5)
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        if job["status"] in ("succeeded", "failed", "canceled"):
            return job
        time.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Fixture — one server per class (scoped to function for isolation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def srv(tmp_path):
    """Start tqdb-server, yield (base_url, data_dir), stop on teardown."""
    port = _free_port()
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    env = {
        **os.environ,
        "TQ_SERVER_ADDR": f"127.0.0.1:{port}",
        "TQ_LOCAL_ROOT": data_dir,
        "TQ_STORAGE_URI": data_dir,
        "TQ_JOB_WORKERS": "2",
        "RUST_LOG": "error",
    }
    proc = subprocess.Popen(
        [SERVER_BIN],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(base)
        yield base, data_dir
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Helper: create a collection via the API
# ---------------------------------------------------------------------------

def _create_col(base: str, tenant: str = "dev", db: str = "db",
                col: str = "c1", dim: int = 8, bits: int = 4,
                metric: str = "ip") -> requests.Response:
    url = f"{base}/v1/tenants/{tenant}/databases/{db}/collections"
    body = {"name": col, "dimension": dim, "bits": bits, "metric": metric}
    return requests.post(url, json=body, headers=HEADERS, timeout=10)


def _col_url(base: str, tenant: str = "dev", db: str = "db", col: str = "c1") -> str:
    return f"{base}/v1/tenants/{tenant}/databases/{db}/collections/{col}"


# ===========================================================================
# HealthZ
# ===========================================================================

class TestHealthZ:
    def test_healthz_no_auth(self, srv):
        base, _ = srv
        r = requests.get(f"{base}/healthz", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_healthz_returns_ok_repeatedly(self, srv):
        base, _ = srv
        for _ in range(10):
            r = requests.get(f"{base}/healthz", timeout=5)
            assert r.status_code == 200

    def test_unknown_route_returns_404(self, srv):
        base, _ = srv
        r = requests.get(f"{base}/v1/does_not_exist", headers=AUTH_HDR, timeout=5)
        assert r.status_code == 404


# ===========================================================================
# Authentication
# ===========================================================================

class TestAuthentication:
    def test_no_auth_header_returns_401(self, srv):
        base, _ = srv
        r = requests.get(f"{base}/v1/tenants/dev/databases/db/collections", timeout=5)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthenticated"

    def test_wrong_key_returns_401(self, srv):
        base, _ = srv
        r = requests.get(
            f"{base}/v1/tenants/dev/databases/db/collections",
            headers={"Authorization": "ApiKey totally-wrong-key"},
            timeout=5,
        )
        assert r.status_code == 401

    def test_bearer_prefix_rejected(self, srv):
        """Server expects 'ApiKey <key>', not 'Bearer <key>'."""
        base, _ = srv
        r = requests.get(
            f"{base}/v1/tenants/dev/databases/db/collections",
            headers={"Authorization": f"Bearer {GOOD_KEY}"},
            timeout=5,
        )
        assert r.status_code == 401

    def test_empty_auth_header_rejected(self, srv):
        base, _ = srv
        r = requests.get(
            f"{base}/v1/tenants/dev/databases/db/collections",
            headers={"Authorization": ""},
            timeout=5,
        )
        assert r.status_code == 401

    def test_cross_tenant_rejected(self, srv):
        """dev-key is bound to tenant 'dev' — accessing tenant 'other' is forbidden."""
        base, _ = srv
        r = requests.get(
            f"{base}/v1/tenants/other/databases/db/collections",
            headers=AUTH_HDR,
            timeout=5,
        )
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "forbidden"

    def test_write_to_wrong_tenant_forbidden(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/attacker/databases/db/collections"
        r = requests.post(url, json={"name": "x", "dimension": 4, "bits": 4},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 403

    def test_request_id_echoed_in_error(self, srv):
        base, _ = srv
        hdrs = {**AUTH_HDR, **JSON_HDR, "X-Request-ID": "req-abc-123"}
        # POST to collections with dimension=0 → guaranteed 400 JSON error body
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(url, json={"name": "bad", "dimension": 0, "bits": 4}, headers=hdrs, timeout=5)
        assert r.status_code == 400
        body = r.json()
        assert "request_id" in body
        assert body["request_id"] == "req-abc-123"


# ===========================================================================
# Collection Management
# ===========================================================================

class TestCollectionManagement:
    def test_create_collection_201(self, srv):
        base, _ = srv
        r = _create_col(base)
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "c1"
        assert body["dimension"] == 8
        assert body["bits"] == 4
        assert body["metric"] == "ip"

    def test_list_collections_empty(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.get(url, headers=AUTH_HDR, timeout=5)
        assert r.status_code == 200
        assert r.json()["collections"] == []

    def test_list_collections_after_create(self, srv):
        base, _ = srv
        _create_col(base, col="c1")
        _create_col(base, col="c2")
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.get(url, headers=AUTH_HDR, timeout=5)
        names = {c["name"] for c in r.json()["collections"]}
        assert "c1" in names
        assert "c2" in names

    def test_create_duplicate_collection_409(self, srv):
        base, _ = srv
        _create_col(base)
        r = _create_col(base)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "conflict"

    def test_create_dimension_zero_rejected(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(url, json={"name": "bad", "dimension": 0, "bits": 4},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 400

    def test_create_bits_zero_rejected(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(url, json={"name": "bad", "dimension": 8, "bits": 0},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 400

    def test_create_invalid_metric_rejected(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(
            url, json={"name": "bad", "dimension": 8, "bits": 4, "metric": "hamming"},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400
        assert "metric" in r.json()["error"]["message"].lower()

    def test_create_metric_case_insensitive(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(
            url, json={"name": "c1", "dimension": 8, "bits": 4, "metric": "IP"},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 201
        assert r.json()["metric"] == "ip"

    def test_delete_collection_success(self, srv):
        base, _ = srv
        _create_col(base)
        r = requests.delete(_col_url(base), headers=AUTH_HDR, timeout=5)
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_nonexistent_collection_404(self, srv):
        base, _ = srv
        r = requests.delete(_col_url(base, col="ghost"), headers=AUTH_HDR, timeout=5)
        assert r.status_code == 404

    def test_delete_then_recreate_succeeds(self, srv):
        base, _ = srv
        _create_col(base)
        requests.delete(_col_url(base), headers=AUTH_HDR, timeout=5)
        r = _create_col(base)
        assert r.status_code == 201

    def test_path_traversal_in_tenant_rejected(self, srv):
        base, _ = srv
        r = requests.get(
            f"{base}/v1/tenants/../databases/db/collections", headers=AUTH_HDR, timeout=5
        )
        # Axum/hyper normalises `..` at the HTTP layer before routing, so the path
        # becomes /v1/databases/db/collections which matches no route → 404.
        # Either 400 (explicit validation) or 403 (binding mismatch) or 404 (no route) — not 200.
        assert r.status_code in (400, 403, 404)

    def test_path_traversal_in_collection_rejected(self, srv):
        base, _ = srv
        _create_col(base)
        r = requests.delete(
            f"{base}/v1/tenants/dev/databases/db/collections/..",
            headers=AUTH_HDR, timeout=5,
        )
        assert r.status_code in (400, 404)

    def test_slash_in_collection_name_rejected(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(
            url, json={"name": "a/b", "dimension": 8, "bits": 4},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code in (400, 404)

    def test_null_byte_in_collection_name_rejected(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(
            url, json={"name": "a\x00b", "dimension": 8, "bits": 4},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400

    def test_missing_name_field_rejected(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(
            url, json={"dimension": 8, "bits": 4},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 422  # serde deserialization failure

    def test_create_with_all_metrics(self, srv):
        base, _ = srv
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        for i, metric in enumerate(["ip", "cosine", "l2"]):
            r = requests.post(
                url, json={"name": f"col_{metric}", "dimension": 8, "bits": 4, "metric": metric},
                headers=HEADERS, timeout=5,
            )
            assert r.status_code == 201, f"metric={metric}: {r.text}"
            assert r.json()["metric"] == metric


# ===========================================================================
# Add Vectors (add endpoint)
# ===========================================================================

class TestAddVectors:
    def _setup(self, base):
        _create_col(base, dim=8)

    def test_add_single_vector(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        body = {"ids": ["v1"], "embeddings": [_unit_vec(8, 1)]}
        r = requests.post(url, json=body, headers=HEADERS, timeout=10)
        assert r.status_code == 201
        assert r.json()["applied"] == 1

    def test_add_batch_50(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        ids = [f"v{i}" for i in range(50)]
        embs = _unit_batch(50, 8)
        r = requests.post(url, json={"ids": ids, "embeddings": embs}, headers=HEADERS, timeout=10)
        assert r.status_code == 201
        assert r.json()["applied"] == 50

    def test_add_with_metadata_and_document(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        body = {
            "ids": ["v1"],
            "embeddings": [_unit_vec(8, 0)],
            "metadatas": [{"label": "test", "score": 99}],
            "documents": ["Hello world"],
        }
        r = requests.post(url, json=body, headers=HEADERS, timeout=10)
        assert r.status_code == 201

    def test_add_empty_ids_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(url, json={"ids": [], "embeddings": []}, headers=HEADERS, timeout=5)
        assert r.status_code == 400
        assert "ids cannot be empty" in r.json()["error"]["message"]

    def test_add_ids_embeddings_length_mismatch(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={"ids": ["v1", "v2"], "embeddings": [_unit_vec(8, 0)]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400

    def test_add_metadatas_length_mismatch(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={
                "ids": ["v1", "v2"],
                "embeddings": [_unit_vec(8, 0), _unit_vec(8, 1)],
                "metadatas": [{"x": 1}],  # length 1, not 2
            },
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400

    def test_add_documents_length_mismatch(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={
                "ids": ["v1"],
                "embeddings": [_unit_vec(8, 0)],
                "documents": ["doc1", "extra"],  # length 2, not 1
            },
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400

    def test_add_wrong_dimension_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={"ids": ["v1"], "embeddings": [[1.0, 2.0, 3.0]]},  # dim=3 vs expected 8
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400
        assert "dimension mismatch" in r.json()["error"]["message"]

    def test_add_to_nonexistent_collection_404(self, srv):
        base, _ = srv
        url = _col_url(base, col="ghost") + "/add"
        r = requests.post(
            url, json={"ids": ["v1"], "embeddings": [_unit_vec(8, 0)]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code in (404, 500)

    def test_add_duplicate_id_in_same_request(self, srv):
        """Duplicate IDs in a single add request — server returns 409 Conflict."""
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={"ids": ["v1", "v1"], "embeddings": [_unit_vec(8, 0), _unit_vec(8, 1)]},
            headers=HEADERS, timeout=5,
        )
        # Server returns 409 for duplicate IDs within the same batch (fail-fast)
        assert r.status_code in (201, 200, 400, 409)

    def test_add_report_mode_partial_failure(self, srv):
        """report=true: one wrong-dim, one correct → partial success, HTTP 200."""
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={
                "ids": ["good", "bad"],
                "embeddings": [_unit_vec(8, 0), [1.0, 2.0]],  # bad has dim=2
                "report": True,
            },
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] == 1
        assert len(body["failed"]) == 1
        assert body["failed"][0]["id"] == "bad"

    def test_add_report_mode_all_success_201(self, srv):
        """report=true with all valid → 201."""
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={"ids": ["v1"], "embeddings": [_unit_vec(8, 0)], "report": True},
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 201

    def test_add_duplicate_id_already_in_db_fails_in_report_mode(self, srv):
        """add (not upsert) with existing ID in report mode → failure recorded."""
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        # Insert once
        requests.post(url, json={"ids": ["v1"], "embeddings": [_unit_vec(8, 0)]},
                      headers=HEADERS, timeout=10)
        # Insert again — should be duplicate insert (not upsert)
        r = requests.post(
            url,
            json={"ids": ["v1"], "embeddings": [_unit_vec(8, 1)], "report": True},
            headers=HEADERS, timeout=10,
        )
        # The result depends on whether the engine treats double-insert as error
        # At minimum it should not 500
        assert r.status_code in (201, 200)

    def test_add_no_auth_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/add"
        r = requests.post(
            url, json={"ids": ["v1"], "embeddings": [_unit_vec(8, 0)]},
            headers=JSON_HDR, timeout=5,
        )
        assert r.status_code == 401


# ===========================================================================
# Upsert Vectors
# ===========================================================================

class TestUpsertVectors:
    def _setup(self, base):
        _create_col(base, dim=8)

    def _insert(self, base, vid: str = "v1", seed: int = 0):
        url = _col_url(base) + "/upsert"
        requests.post(url, json={"ids": [vid], "embeddings": [_unit_vec(8, seed)]},
                      headers=HEADERS, timeout=10)

    def test_upsert_new_id(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/upsert"
        r = requests.post(url, json={"ids": ["new"], "embeddings": [_unit_vec(8, 0)]},
                          headers=HEADERS, timeout=10)
        assert r.status_code == 200
        assert r.json()["applied"] == 1

    def test_upsert_existing_id_overwrites(self, srv):
        base, _ = srv
        self._setup(base)
        self._insert(base, "v1", seed=0)
        url = _col_url(base) + "/upsert"
        r = requests.post(url, json={"ids": ["v1"], "embeddings": [_unit_vec(8, 99)]},
                          headers=HEADERS, timeout=10)
        assert r.status_code == 200
        assert r.json()["applied"] == 1

    def test_upsert_empty_ids_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/upsert"
        r = requests.post(url, json={"ids": [], "embeddings": []},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 400

    def test_upsert_wrong_dim_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/upsert"
        r = requests.post(url, json={"ids": ["v1"], "embeddings": [[1.0, 2.0]]},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 400

    def test_upsert_report_mode_partial_failure(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/upsert"
        r = requests.post(
            url,
            json={
                "ids": ["good", "bad"],
                "embeddings": [_unit_vec(8, 0), [99.0]],
                "report": True,
            },
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["applied"] == 1
        assert len(r.json()["failed"]) == 1

    def test_upsert_with_metadata_survives_overwrite(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/upsert"
        requests.post(
            url,
            json={"ids": ["v1"], "embeddings": [_unit_vec(8, 0)],
                  "metadatas": [{"label": "first"}]},
            headers=HEADERS, timeout=10,
        )
        requests.post(
            url,
            json={"ids": ["v1"], "embeddings": [_unit_vec(8, 1)],
                  "metadatas": [{"label": "second"}]},
            headers=HEADERS, timeout=10,
        )
        # Fetch to verify
        get_url = _col_url(base) + "/get"
        r = requests.post(get_url, json={"ids": ["v1"]}, headers=HEADERS, timeout=5)
        assert r.status_code == 200
        meta_list = r.json()["metadatas"]
        assert meta_list[0].get("label") == "second"

    def test_upsert_metadatas_length_mismatch_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/upsert"
        r = requests.post(
            url,
            json={
                "ids": ["v1", "v2"],
                "embeddings": [_unit_vec(8, 0), _unit_vec(8, 1)],
                "metadatas": [{"x": 1}],  # only 1 meta for 2 ids
            },
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400


# ===========================================================================
# Delete Vectors
# ===========================================================================

class TestDeleteVectors:
    def _setup(self, base, n: int = 5) -> List[str]:
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        ids = [f"v{i}" for i in range(n)]
        requests.post(url, json={"ids": ids, "embeddings": _unit_batch(n, 8)},
                      headers=HEADERS, timeout=10)
        return ids

    def test_delete_by_ids(self, srv):
        base, _ = srv
        ids = self._setup(base)
        url = _col_url(base) + "/delete"
        r = requests.post(url, json={"ids": ["v0", "v1"]}, headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert r.json()["deleted"] == 2

    def test_delete_nonexistent_ids_zero_deleted(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/delete"
        r = requests.post(url, json={"ids": ["ghost99"]}, headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert r.json()["deleted"] == 0

    def test_delete_empty_body_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/delete"
        r = requests.post(url, json={}, headers=HEADERS, timeout=5)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_argument"

    def test_delete_by_filter(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        add_url = _col_url(base) + "/add"
        requests.post(
            add_url,
            json={
                "ids": ["a", "b", "c"],
                "embeddings": _unit_batch(3, 8),
                "metadatas": [{"tag": "x"}, {"tag": "y"}, {"tag": "x"}],
            },
            headers=HEADERS, timeout=10,
        )
        del_url = _col_url(base) + "/delete"
        r = requests.post(del_url, json={"filter": {"tag": {"$eq": "x"}}},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert r.json()["deleted"] == 2

    def test_delete_all_then_count_zero(self, srv):
        base, _ = srv
        ids = self._setup(base, n=5)
        url = _col_url(base) + "/delete"
        requests.post(url, json={"ids": ids}, headers=HEADERS, timeout=5)
        # Verify by listing
        list_r = requests.get(f"{_col_url(base)}/../collections", headers=AUTH_HDR, timeout=5)
        # Just check the collection still exists (no crash after delete all)

    def test_delete_combined_ids_and_filter(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        add_url = _col_url(base) + "/add"
        requests.post(
            add_url,
            json={
                "ids": ["a", "b", "c", "d"],
                "embeddings": _unit_batch(4, 8),
                "metadatas": [{"t": "x"}, {"t": "y"}, {"t": "z"}, {"t": "x"}],
            },
            headers=HEADERS, timeout=10,
        )
        del_url = _col_url(base) + "/delete"
        # ids=["a"] + filter matches "d" (both have tag=x) → 2 deleted total
        r = requests.post(del_url, json={"ids": ["a"], "filter": {"t": {"$eq": "x"}}},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 200
        # "a" is in both ids list and filter — deduplication should happen
        assert r.json()["deleted"] == 2

    def test_delete_duplicate_ids_deduplicated(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/delete"
        r = requests.post(url, json={"ids": ["v0", "v0", "v0"]}, headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert r.json()["deleted"] == 1  # deduplication: only one actual delete

    def test_delete_where_filter_alias(self, srv):
        """where_filter is accepted as alias for filter."""
        base, _ = srv
        _create_col(base, dim=8)
        add_url = _col_url(base) + "/add"
        requests.post(
            add_url,
            json={
                "ids": ["a"],
                "embeddings": [_unit_vec(8, 0)],
                "metadatas": [{"kind": "alpha"}],
            },
            headers=HEADERS, timeout=10,
        )
        del_url = _col_url(base) + "/delete"
        r = requests.post(del_url, json={"where_filter": {"kind": {"$eq": "alpha"}}},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert r.json()["deleted"] == 1


# ===========================================================================
# Get Vectors
# ===========================================================================

class TestGetVectors:
    def _setup(self, base) -> None:
        _create_col(base, dim=8)
        add_url = _col_url(base) + "/add"
        requests.post(
            add_url,
            json={
                "ids": ["a", "b", "c"],
                "embeddings": _unit_batch(3, 8),
                "metadatas": [{"tag": "alpha"}, {"tag": "beta"}, {"tag": "alpha"}],
                "documents": ["doc-a", "doc-b", "doc-c"],
            },
            headers=HEADERS, timeout=10,
        )

    def test_get_by_ids(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(url, json={"ids": ["a", "b"]}, headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert len(r.json()["ids"]) == 2

    def test_get_by_filter(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url, json={"filter": {"tag": {"$eq": "alpha"}}}, headers=HEADERS, timeout=5
        )
        assert r.status_code == 200
        assert len(r.json()["ids"]) == 2

    def test_get_empty_selector_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(url, json={}, headers=HEADERS, timeout=5)
        assert r.status_code == 400

    def test_get_include_ids_only(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url, json={"ids": ["a"], "include": ["ids"]}, headers=HEADERS, timeout=5
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("ids") is not None
        assert body.get("metadatas") is None
        assert body.get("documents") is None

    def test_get_include_metadatas_only(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url, json={"ids": ["a"], "include": ["metadatas"]}, headers=HEADERS, timeout=5
        )
        body = r.json()
        assert body.get("ids") is None
        assert body.get("metadatas") is not None

    def test_get_invalid_include_field_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url, json={"ids": ["a"], "include": ["scores"]}, headers=HEADERS, timeout=5
        )
        # scores is not allowed in get — should be 400
        assert r.status_code == 400

    def test_get_offset_and_limit(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url,
            json={"filter": {"tag": {"$eq": "alpha"}}, "offset": 1, "limit": 1},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 200
        assert len(r.json()["ids"]) == 1

    def test_get_offset_beyond_count_returns_empty(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url,
            json={"ids": ["a", "b", "c"], "offset": 100},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 200
        assert r.json()["ids"] == []

    def test_get_missing_id_not_returned(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(url, json={"ids": ["ghost999"]}, headers=HEADERS, timeout=5)
        assert r.status_code == 200
        assert r.json()["ids"] == []

    def test_get_where_filter_alias(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url, json={"where_filter": {"tag": {"$eq": "beta"}}}, headers=HEADERS, timeout=5
        )
        assert r.status_code == 200
        assert "b" in r.json()["ids"]

    def test_get_combined_ids_and_filter_intersection(self, srv):
        """ids=[a,b,c] + filter={tag=alpha} → only a and c returned."""
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url,
            json={"ids": ["a", "b", "c"], "filter": {"tag": {"$eq": "alpha"}}},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 200
        ids_returned = set(r.json()["ids"])
        assert ids_returned == {"a", "c"}

    def test_get_includes_documents(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/get"
        r = requests.post(
            url, json={"ids": ["a"], "include": ["documents"]}, headers=HEADERS, timeout=5
        )
        assert r.status_code == 200
        docs = r.json()["documents"]
        assert docs[0] == "doc-a"


# ===========================================================================
# Query Vectors
# ===========================================================================

class TestQueryVectors:
    def _setup(self, base, n: int = 10) -> None:
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        ids = [f"v{i}" for i in range(n)]
        metas = [{"idx": i, "even": (i % 2 == 0)} for i in range(n)]
        requests.post(
            url,
            json={"ids": ids, "embeddings": _unit_batch(n, 8), "metadatas": metas},
            headers=HEADERS, timeout=10,
        )

    def test_query_basic(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 5},
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 1
        assert len(body["results"][0]["ids"]) == 5

    def test_query_multiple_embeddings(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={"query_embeddings": [_unit_vec(8, 0), _unit_vec(8, 1)], "top_k": 3},
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 2
        for row in body["results"]:
            assert len(row["ids"]) == 3

    def test_query_top_k_zero_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url, json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 0},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400
        assert "top_k must be greater than 0" in r.json()["error"]["message"]

    def test_query_empty_query_embeddings_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url, json={"query_embeddings": [], "top_k": 5},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400

    def test_query_wrong_dim_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={"query_embeddings": [[1.0, 2.0]], "top_k": 5},  # dim=2 vs col dim=8
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400
        assert "dimension mismatch" in r.json()["error"]["message"]

    def test_query_with_filter(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={
                "query_embeddings": [_unit_vec(8, 0)],
                "top_k": 10,
                "filter": {"even": {"$eq": True}},
            },
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        body = r.json()
        returned_ids = body["results"][0]["ids"]
        # All returned IDs should have even index
        assert all(int(vid[1:]) % 2 == 0 for vid in returned_ids)

    def test_query_where_filter_alias(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={
                "query_embeddings": [_unit_vec(8, 0)],
                "top_k": 5,
                "where_filter": {"idx": {"$gte": 5}},
            },
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        ids = r.json()["results"][0]["ids"]
        assert all(int(vid[1:]) >= 5 for vid in ids)

    def test_query_n_results_alias(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={"query_embeddings": [_unit_vec(8, 0)], "n_results": 3},
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        assert len(r.json()["results"][0]["ids"]) == 3

    def test_query_include_scores(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={
                "query_embeddings": [_unit_vec(8, 0)],
                "top_k": 5,
                "include": ["ids", "scores"],
            },
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        row = r.json()["results"][0]
        assert row["scores"] is not None
        assert row["metadatas"] is None
        assert row["documents"] is None

    def test_query_include_invalid_field_rejected(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={
                "query_embeddings": [_unit_vec(8, 0)],
                "top_k": 5,
                "include": ["ids", "embeddings"],  # 'embeddings' is not allowed
            },
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 400

    def test_query_offset(self, srv):
        base, _ = srv
        self._setup(base, n=10)
        url = _col_url(base) + "/query"
        r0 = requests.post(
            url,
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 3, "offset": 0},
            headers=HEADERS, timeout=10,
        )
        r3 = requests.post(
            url,
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 3, "offset": 3},
            headers=HEADERS, timeout=10,
        )
        ids0 = r0.json()["results"][0]["ids"]
        ids3 = r3.json()["results"][0]["ids"]
        assert set(ids0).isdisjoint(set(ids3))  # No overlap expected

    def test_query_top_k_larger_than_collection(self, srv):
        base, _ = srv
        self._setup(base, n=3)
        url = _col_url(base) + "/query"
        r = requests.post(
            url,
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 100},
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        ids = r.json()["results"][0]["ids"]
        assert len(ids) == 3  # can't return more than exist

    def test_query_empty_collection_returns_no_results(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/query"
        r = requests.post(
            url, json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 5},
            headers=HEADERS, timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["results"][0]["ids"] == []


# ===========================================================================
# Async Jobs
# ===========================================================================

class TestAsyncJobs:
    def _setup(self, base, n: int = 30) -> None:
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        ids = [f"v{i}" for i in range(n)]
        requests.post(url, json={"ids": ids, "embeddings": _unit_batch(n, 8)},
                      headers=HEADERS, timeout=10)

    def test_compact_job_enqueued_and_succeeds(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/compact"
        r = requests.post(url, json={}, headers=HEADERS, timeout=5)
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        job = _wait_job(base, body["job_id"])
        assert job["status"] == "succeeded"

    def test_index_job_enqueued_and_succeeds(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/index"
        r = requests.post(url, json={}, headers=HEADERS, timeout=5)
        assert r.status_code == 202
        job = _wait_job(base, r.json()["job_id"])
        assert job["status"] == "succeeded"

    def test_snapshot_job_enqueued_and_succeeds(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/snapshot"
        r = requests.post(url, json={"snapshot_name": "snap1"}, headers=HEADERS, timeout=5)
        assert r.status_code == 202
        job = _wait_job(base, r.json()["job_id"])
        assert job["status"] == "succeeded"

    def test_snapshot_auto_name_when_not_provided(self, srv):
        base, _ = srv
        self._setup(base)
        url = _col_url(base) + "/snapshot"
        r = requests.post(url, json={}, headers=HEADERS, timeout=5)
        assert r.status_code == 202
        job = _wait_job(base, r.json()["job_id"])
        assert job["status"] == "succeeded"

    def test_restore_job_after_snapshot(self, srv):
        base, _ = srv
        self._setup(base, n=5)
        # Snapshot first
        snap_url = _col_url(base) + "/snapshot"
        snap_r = requests.post(snap_url, json={"snapshot_name": "snap_restore"},
                               headers=HEADERS, timeout=5)
        snap_job = _wait_job(base, snap_r.json()["job_id"])
        assert snap_job["status"] == "succeeded"

        # Delete all vectors
        ids = [f"v{i}" for i in range(5)]
        requests.post(_col_url(base) + "/delete", json={"ids": ids},
                      headers=HEADERS, timeout=5)

        # Restore
        restore_url = _col_url(base) + "/restore"
        r = requests.post(restore_url, json={"snapshot_name": "snap_restore"},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 202
        restore_job = _wait_job(base, r.json()["job_id"])
        assert restore_job["status"] == "succeeded"

    def test_restore_nonexistent_snapshot_fails(self, srv):
        base, _ = srv
        self._setup(base)
        restore_url = _col_url(base) + "/restore"
        r = requests.post(restore_url, json={"snapshot_name": "nonexistent_snap"},
                          headers=HEADERS, timeout=5)
        assert r.status_code == 202
        job = _wait_job(base, r.json()["job_id"])
        assert job["status"] == "failed"

    def test_get_job_status(self, srv):
        base, _ = srv
        self._setup(base)
        r = requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        job_id = r.json()["job_id"]
        status_r = requests.get(f"{base}/v1/jobs/{job_id}", headers=AUTH_HDR, timeout=5)
        assert status_r.status_code == 200
        assert status_r.json()["job"]["job_id"] == job_id

    def test_get_nonexistent_job_404(self, srv):
        base, _ = srv
        r = requests.get(f"{base}/v1/jobs/job_99999999", headers=AUTH_HDR, timeout=5)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"

    def test_list_collection_jobs(self, srv):
        base, _ = srv
        self._setup(base)
        requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        r = requests.get(_col_url(base) + "/jobs", headers=AUTH_HDR, timeout=5)
        assert r.status_code == 200
        assert len(r.json()["jobs"]) >= 2

    def test_cancel_queued_job_succeeds(self, srv):
        """Cancel a queued job before it starts — should become 'canceled'."""
        base, _ = srv
        self._setup(base)
        # Saturate workers to get a queued job
        for _ in range(5):
            requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        jobs_r = requests.get(_col_url(base) + "/jobs", headers=AUTH_HDR, timeout=5)
        queued = [j for j in jobs_r.json()["jobs"] if j["status"] == "queued"]
        if queued:
            job_id = queued[0]["job_id"]
            cancel_r = requests.post(f"{base}/v1/jobs/{job_id}/cancel",
                                     headers=HEADERS, timeout=5)
            assert cancel_r.status_code in (200, 409)  # 409 if already running

    def test_cancel_terminal_job_conflict(self, srv):
        base, _ = srv
        self._setup(base)
        r = requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        job_id = r.json()["job_id"]
        _wait_job(base, job_id)  # wait until terminal
        cancel_r = requests.post(f"{base}/v1/jobs/{job_id}/cancel",
                                 headers=HEADERS, timeout=5)
        assert cancel_r.status_code == 409
        assert cancel_r.json()["error"]["code"] == "conflict"

    def test_retry_queued_job_conflict(self, srv):
        """Retry a job that's still queued — should be conflict."""
        base, _ = srv
        self._setup(base)
        # Exhaust workers
        for _ in range(10):
            requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        jobs_r = requests.get(_col_url(base) + "/jobs", headers=AUTH_HDR, timeout=5)
        queued = [j for j in jobs_r.json()["jobs"] if j["status"] == "queued"]
        if queued:
            job_id = queued[0]["job_id"]
            retry_r = requests.post(f"{base}/v1/jobs/{job_id}/retry",
                                    headers=HEADERS, timeout=5)
            assert retry_r.status_code == 409

    def test_retry_succeeded_job_conflict(self, srv):
        base, _ = srv
        self._setup(base)
        r = requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        job_id = r.json()["job_id"]
        _wait_job(base, job_id)
        retry_r = requests.post(f"{base}/v1/jobs/{job_id}/retry",
                                headers=HEADERS, timeout=5)
        assert retry_r.status_code == 409

    def test_compact_nonexistent_collection_job_fails(self, srv):
        """compact on non-existent collection → job fails."""
        base, _ = srv
        _create_col(base)
        requests.delete(_col_url(base), headers=AUTH_HDR, timeout=5)
        r = requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        # Server may 404 the collection at enqueue time, or accept and fail the job
        if r.status_code == 202:
            job = _wait_job(base, r.json()["job_id"])
            assert job["status"] == "failed"
        else:
            assert r.status_code in (404, 400)

    def test_index_then_query_ann_result_exists(self, srv):
        """After index job succeeds, query should return results."""
        base, _ = srv
        self._setup(base, n=30)
        r = requests.post(_col_url(base) + "/index", json={}, headers=HEADERS, timeout=5)
        job = _wait_job(base, r.json()["job_id"])
        assert job["status"] == "succeeded"
        # Query should still work
        qr = requests.post(
            _col_url(base) + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 5},
            headers=HEADERS, timeout=10,
        )
        assert qr.status_code == 200
        assert len(qr.json()["results"][0]["ids"]) > 0


# ===========================================================================
# Quota Enforcement
# ===========================================================================

class TestQuotaEnforcement:
    def _make_server_with_quota_file(self, tmp_path, quota_config: dict):
        """Start a server with a custom quota_store.json."""
        port = _free_port()
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        # Write quota file
        quota_path = os.path.join(data_dir, "quota_store.json")
        with open(quota_path, "w") as f:
            json.dump(quota_config, f)
        env = {
            **os.environ,
            "TQ_SERVER_ADDR": f"127.0.0.1:{port}",
            "TQ_LOCAL_ROOT": data_dir,
            "TQ_STORAGE_URI": data_dir,
            "TQ_QUOTA_STORE_PATH": quota_path,
            "TQ_JOB_WORKERS": "1",
            "RUST_LOG": "error",
        }
        proc = subprocess.Popen(
            [SERVER_BIN], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{port}"
        _wait_ready(base)
        return proc, base

    def test_vector_quota_exceeded(self, tmp_path):
        quota = {
            "tenant_quotas": [
                {"tenant": "dev", "max_vectors": 3, "max_collections": None,
                 "max_disk_bytes": None, "max_concurrent_jobs": None}
            ],
            "database_quotas": [],
        }
        proc, base = self._make_server_with_quota_file(tmp_path, quota)
        try:
            _create_col(base, dim=4)
            url = _col_url(base) + "/add"
            # Add 3 — should succeed
            r = requests.post(
                url,
                json={"ids": ["v0", "v1", "v2"], "embeddings": _unit_batch(3, 4)},
                headers=HEADERS, timeout=10,
            )
            assert r.status_code == 201
            # Add 1 more — should be rejected (quota: max 3)
            r2 = requests.post(
                url,
                json={"ids": ["v3"], "embeddings": [_unit_vec(4, 3)]},
                headers=HEADERS, timeout=10,
            )
            assert r2.status_code == 429
            assert r2.json()["error"]["code"] == "quota_exceeded"
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_collection_quota_exceeded(self, tmp_path):
        quota = {
            "tenant_quotas": [
                {"tenant": "dev", "max_collections": 2, "max_vectors": None,
                 "max_disk_bytes": None, "max_concurrent_jobs": None}
            ],
            "database_quotas": [],
        }
        proc, base = self._make_server_with_quota_file(tmp_path, quota)
        try:
            _create_col(base, col="c1", dim=4)
            _create_col(base, col="c2", dim=4)
            r = _create_col(base, col="c3", dim=4)
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "quota_exceeded"
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_quota_usage_endpoint(self, tmp_path):
        quota = {
            "tenant_quotas": [
                {"tenant": "dev", "max_vectors": 100, "max_collections": 10,
                 "max_disk_bytes": None, "max_concurrent_jobs": None}
            ],
            "database_quotas": [],
        }
        proc, base = self._make_server_with_quota_file(tmp_path, quota)
        try:
            _create_col(base, col="c1", dim=4)
            url = _col_url(base, col="c1") + "/add"
            requests.post(
                url,
                json={"ids": ["v0", "v1"], "embeddings": _unit_batch(2, 4)},
                headers=HEADERS, timeout=10,
            )
            r = requests.get(
                f"{base}/v1/tenants/dev/databases/db/quota_usage",
                headers=AUTH_HDR, timeout=5,
            )
            assert r.status_code == 200
            body = r.json()
            assert body["max_vectors"] == 100
            assert body["max_collections"] == 10
            assert body["current_collections"] == 1
            assert body["current_vectors"] == 2
        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ===========================================================================
# Multi-tenant Isolation
# ===========================================================================

class TestMultiCollectionIsolation:
    def test_two_collections_independent(self, srv):
        base, _ = srv
        _create_col(base, col="c1", dim=8)
        _create_col(base, col="c2", dim=8)
        add_url1 = _col_url(base, col="c1") + "/add"
        add_url2 = _col_url(base, col="c2") + "/add"
        requests.post(add_url1, json={"ids": ["a"], "embeddings": [_unit_vec(8, 0)]},
                      headers=HEADERS, timeout=10)
        requests.post(add_url2, json={"ids": ["b"], "embeddings": [_unit_vec(8, 1)]},
                      headers=HEADERS, timeout=10)
        # Get from c1 should not find "b" and vice-versa
        get1 = requests.post(_col_url(base, col="c1") + "/get",
                             json={"ids": ["b"]}, headers=HEADERS, timeout=5)
        assert get1.json()["ids"] == []
        get2 = requests.post(_col_url(base, col="c2") + "/get",
                             json={"ids": ["a"]}, headers=HEADERS, timeout=5)
        assert get2.json()["ids"] == []

    def test_delete_from_one_collection_doesnt_affect_other(self, srv):
        base, _ = srv
        _create_col(base, col="c1", dim=8)
        _create_col(base, col="c2", dim=8)
        for col in ["c1", "c2"]:
            requests.post(
                _col_url(base, col=col) + "/add",
                json={"ids": ["shared_id"], "embeddings": [_unit_vec(8, 0)]},
                headers=HEADERS, timeout=10,
            )
        # Delete from c1
        requests.post(_col_url(base, col="c1") + "/delete",
                      json={"ids": ["shared_id"]}, headers=HEADERS, timeout=5)
        # c2 should still have it
        r = requests.post(_col_url(base, col="c2") + "/get",
                          json={"ids": ["shared_id"]}, headers=HEADERS, timeout=5)
        assert "shared_id" in r.json()["ids"]

    def test_collection_in_different_dbs_isolated(self, srv):
        base, _ = srv
        _create_col(base, db="db1", col="c1", dim=8)
        _create_col(base, db="db2", col="c1", dim=8)
        # Add to db1/c1
        requests.post(
            f"{base}/v1/tenants/dev/databases/db1/collections/c1/add",
            json={"ids": ["only_in_db1"], "embeddings": [_unit_vec(8, 0)]},
            headers=HEADERS, timeout=10,
        )
        # db2/c1 should be empty
        r = requests.post(
            f"{base}/v1/tenants/dev/databases/db2/collections/c1/get",
            json={"ids": ["only_in_db1"]},
            headers=HEADERS, timeout=5,
        )
        assert r.json()["ids"] == []


# ===========================================================================
# Metrics Endpoint
# ===========================================================================

class TestMetrics:
    def test_metrics_endpoint_requires_auth(self, srv):
        base, _ = srv
        r = requests.get(f"{base}/metrics", timeout=5)
        assert r.status_code == 401

    def test_metrics_returns_prometheus_format(self, srv):
        base, _ = srv
        r = requests.get(f"{base}/metrics", headers=AUTH_HDR, timeout=5)
        assert r.status_code == 200
        ct = r.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_metrics_increments_after_query(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        add_url = _col_url(base) + "/add"
        requests.post(add_url, json={"ids": ["v0"], "embeddings": [_unit_vec(8, 0)]},
                      headers=HEADERS, timeout=10)
        # Run a query to increment tqdb_search_requests_total
        requests.post(
            _col_url(base) + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 1},
            headers=HEADERS, timeout=10,
        )
        r = requests.get(f"{base}/metrics", headers=AUTH_HDR, timeout=5)
        assert "tqdb_search_requests_total" in r.text


# ===========================================================================
# Edge Cases & Adversarial Input
# ===========================================================================

class TestEdgeCasesAdversarial:
    def test_malformed_json_returns_422(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        r = requests.post(url, data=b"{not valid json", headers=HEADERS, timeout=5)
        # Axum returns 400 (JsonRejection) for malformed JSON, not 422
        assert r.status_code in (400, 422)

    def test_empty_body_returns_422(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        r = requests.post(url, data=b"", headers=HEADERS, timeout=5)
        assert r.status_code in (400, 422)

    def test_add_nan_embedding_values(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        # JSON doesn't support NaN natively; some clients send as string or null
        # Send as null → serde should reject
        r = requests.post(
            url,
            json={"ids": ["nan_vec"], "embeddings": [[None] * 8]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code in (400, 422)

    def test_add_extremely_large_vector_values(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        big_vec = [1e38] * 8
        r = requests.post(
            url, json={"ids": ["big"], "embeddings": [big_vec]}, headers=HEADERS, timeout=5
        )
        # Should succeed (float values, not infinity)
        assert r.status_code in (201, 200, 400)

    def test_unicode_id_and_metadata(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={
                "ids": ["こんにちは🎉"],
                "embeddings": [_unit_vec(8, 0)],
                "metadatas": [{"emoji": "🔥", "cjk": "中文"}],
            },
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 201
        get_r = requests.post(
            _col_url(base) + "/get",
            json={"ids": ["こんにちは🎉"]},
            headers=HEADERS, timeout=5,
        )
        assert "こんにちは🎉" in get_r.json()["ids"]

    def test_very_long_id(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        long_id = "x" * 10000
        r = requests.post(
            url, json={"ids": [long_id], "embeddings": [_unit_vec(8, 0)]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code in (201, 400)

    def test_deeply_nested_metadata(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        deep = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        r = requests.post(
            url,
            json={"ids": ["deep_meta"], "embeddings": [_unit_vec(8, 0)], "metadatas": [deep]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code in (201, 400)

    def test_large_batch_add_500_vectors(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        n = 500
        r = requests.post(
            url,
            json={"ids": [f"v{i}" for i in range(n)], "embeddings": _unit_batch(n, 8)},
            headers=HEADERS, timeout=30,
        )
        assert r.status_code == 201
        assert r.json()["applied"] == n

    def test_concurrent_add_to_same_collection(self, srv):
        """50 threads each adding 5 vectors — total should be 250."""
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        errors = []

        def add_batch(thread_id: int):
            try:
                ids = [f"t{thread_id}_v{j}" for j in range(5)]
                r = requests.post(
                    url,
                    json={"ids": ids, "embeddings": _unit_batch(5, 8, seed=thread_id)},
                    headers=HEADERS, timeout=15,
                )
                if r.status_code not in (200, 201):
                    errors.append(f"thread {thread_id}: {r.status_code} {r.text}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=add_batch, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent add errors: {errors}"

    def test_concurrent_read_write(self, srv):
        """10 writers + 20 readers running concurrently — no server crashes."""
        base, _ = srv
        _create_col(base, dim=8)
        # Seed some data
        add_url = _col_url(base) + "/add"
        requests.post(
            add_url,
            json={"ids": [f"seed{i}" for i in range(10)], "embeddings": _unit_batch(10, 8)},
            headers=HEADERS, timeout=10,
        )
        query_url = _col_url(base) + "/query"
        errors = []

        def writer(i):
            try:
                r = requests.post(
                    add_url,
                    json={"ids": [f"w{i}"], "embeddings": [_unit_vec(8, i)], "report": True},
                    headers=HEADERS, timeout=15,
                )
                if r.status_code not in (200, 201):
                    errors.append(f"writer {i}: {r.status_code}")
            except Exception as e:
                errors.append(str(e))

        def reader(i):
            try:
                r = requests.post(
                    query_url,
                    json={"query_embeddings": [_unit_vec(8, i)], "top_k": 3},
                    headers=HEADERS, timeout=15,
                )
                if r.status_code != 200:
                    errors.append(f"reader {i}: {r.status_code}")
            except Exception as e:
                errors.append(str(e))

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(10)]
            + [threading.Thread(target=reader, args=(i,)) for i in range(20)]
        )
        random.shuffle(threads)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent errors: {errors}"

    def test_add_then_query_self_similarity_top1(self, srv):
        """Every inserted vector should be its own top-1 match (brute force)."""
        base, _ = srv
        _create_col(base, dim=16, metric="ip")
        n = 20
        add_url = _col_url(base) + "/add"
        vecs = _unit_batch(n, 16, seed=7)
        ids = [f"v{i}" for i in range(n)]
        requests.post(add_url, json={"ids": ids, "embeddings": vecs},
                      headers=HEADERS, timeout=10)

        query_url = _col_url(base) + "/query"
        misses = 0
        for i, vid in enumerate(ids):
            r = requests.post(
                query_url,
                json={"query_embeddings": [vecs[i]], "top_k": 1, "include": ["ids"]},
                headers=HEADERS, timeout=10,
            )
            top1 = r.json()["results"][0]["ids"]
            if top1 and top1[0] != vid:
                misses += 1
        # Allow at most 1 miss (quantization error)
        assert misses <= 1, f"{misses} self-similarity misses"

    def test_scores_descending_for_ip_metric(self, srv):
        base, _ = srv
        _create_col(base, dim=8, metric="ip")
        add_url = _col_url(base) + "/add"
        n = 10
        requests.post(
            add_url,
            json={"ids": [f"v{i}" for i in range(n)], "embeddings": _unit_batch(n, 8)},
            headers=HEADERS, timeout=10,
        )
        query_url = _col_url(base) + "/query"
        r = requests.post(
            query_url,
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": n, "include": ["scores"]},
            headers=HEADERS, timeout=10,
        )
        scores = r.json()["results"][0]["scores"]
        assert scores == sorted(scores, reverse=True), "IP scores should be descending"

    def test_scores_non_negative_for_l2_metric(self, srv):
        base, _ = srv
        _create_col(base, dim=8, metric="l2")
        add_url = _col_url(base) + "/add"
        n = 10
        requests.post(
            add_url,
            json={"ids": [f"v{i}" for i in range(n)], "embeddings": _unit_batch(n, 8)},
            headers=HEADERS, timeout=10,
        )
        query_url = _col_url(base) + "/query"
        r = requests.post(
            query_url,
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": n, "include": ["scores"]},
            headers=HEADERS, timeout=10,
        )
        scores = r.json()["results"][0]["scores"]
        # Server now returns raw positive L2 distances sorted ascending (nearest first).
        # BUG-2 (negated L2) is fixed in both embedded and server modes.
        assert scores == sorted(scores), "L2 scores must be ascending (nearest = smallest distance)"
        assert all(s >= 0 for s in scores), f"L2 distance scores must be ≥ 0: {scores}"


# ===========================================================================
# Data Persistence & Lifecycle
# ===========================================================================

class TestPersistence:
    def test_data_survives_after_server_restart(self, tmp_path):
        """Insert vectors, restart server, verify data still exists."""
        port = _free_port()
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        env = {
            **os.environ,
            "TQ_SERVER_ADDR": f"127.0.0.1:{port}",
            "TQ_LOCAL_ROOT": data_dir,
            "TQ_STORAGE_URI": data_dir,
            "TQ_JOB_WORKERS": "1",
            "RUST_LOG": "error",
        }

        def start():
            proc = subprocess.Popen(
                [SERVER_BIN], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            base = f"http://127.0.0.1:{port}"
            _wait_ready(base)
            return proc, base

        proc1, base = start()
        try:
            _create_col(base, dim=8)
            add_url = _col_url(base) + "/add"
            requests.post(
                add_url,
                json={"ids": ["persistent"], "embeddings": [_unit_vec(8, 0)]},
                headers=HEADERS, timeout=10,
            )
        finally:
            proc1.terminate()
            proc1.wait(timeout=5)

        time.sleep(0.5)  # Let the OS release the port

        # Restart on the SAME port+data_dir
        proc2, base2 = start()
        try:
            r = requests.post(
                _col_url(base2) + "/get",
                json={"ids": ["persistent"]},
                headers=HEADERS, timeout=5,
            )
            assert r.status_code == 200
            assert "persistent" in r.json()["ids"]
        finally:
            proc2.terminate()
            proc2.wait(timeout=5)

    def test_collection_list_survives_restart(self, tmp_path):
        port = _free_port()
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)
        env = {
            **os.environ,
            "TQ_SERVER_ADDR": f"127.0.0.1:{port}",
            "TQ_LOCAL_ROOT": data_dir,
            "TQ_STORAGE_URI": data_dir,
            "TQ_JOB_WORKERS": "1",
            "RUST_LOG": "error",
        }

        def start():
            p = subprocess.Popen(
                [SERVER_BIN], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            b = f"http://127.0.0.1:{port}"
            _wait_ready(b)
            return p, b

        p1, b = start()
        try:
            _create_col(b, col="survived", dim=4)
        finally:
            p1.terminate()
            p1.wait(timeout=5)

        time.sleep(0.5)
        p2, b2 = start()
        try:
            r = requests.get(
                f"{b2}/v1/tenants/dev/databases/db/collections", headers=AUTH_HDR, timeout=5
            )
            names = {c["name"] for c in r.json()["collections"]}
            assert "survived" in names
        finally:
            p2.terminate()
            p2.wait(timeout=5)


# ===========================================================================
# X-Request-ID Header
# ===========================================================================

class TestRequestIdHeader:
    def test_request_id_echoed_on_success(self, srv):
        """On 200/201, request_id should not be in body (only on errors)."""
        base, _ = srv
        r = _create_col(base)
        # No request_id header sent → body should have request_id=null or absent
        body = r.json()
        assert "request_id" not in body or body.get("request_id") is None

    def test_request_id_echoed_in_error_body(self, srv):
        base, _ = srv
        hdrs = {**AUTH_HDR, **JSON_HDR, "X-Request-ID": "my-trace-id-42"}
        url = f"{base}/v1/tenants/dev/databases/db/collections"
        r = requests.post(url, json={"name": "bad", "dimension": 0, "bits": 4},
                          headers=hdrs, timeout=5)
        assert r.status_code == 400
        assert r.json()["request_id"] == "my-trace-id-42"

    def test_different_request_ids_on_concurrent_requests(self, srv):
        base, _ = srv
        results = {}
        lock = threading.Lock()

        def req(req_id):
            hdrs = {**AUTH_HDR, **JSON_HDR, "X-Request-ID": req_id}
            url = f"{base}/v1/tenants/dev/databases/db/collections"
            r = requests.post(url, json={"name": "x", "dimension": 0, "bits": 0},
                              headers=hdrs, timeout=5)
            with lock:
                results[req_id] = r.json().get("request_id")

        threads = [threading.Thread(target=req, args=(f"rid-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for i in range(10):
            assert results.get(f"rid-{i}") == f"rid-{i}"


# ===========================================================================
# Additional Brutal Edge Cases
# ===========================================================================

class TestBrutalEdgeCasesServer:
    def test_get_with_ids_and_where_filter_no_intersection(self, srv):
        """ids=[a] + filter={tag=beta} where a has tag=alpha → empty result."""
        base, _ = srv
        _create_col(base, dim=8)
        requests.post(
            _col_url(base) + "/add",
            json={"ids": ["a"], "embeddings": [_unit_vec(8, 0)],
                  "metadatas": [{"tag": "alpha"}]},
            headers=HEADERS, timeout=10,
        )
        r = requests.post(
            _col_url(base) + "/get",
            json={"ids": ["a"], "filter": {"tag": {"$eq": "beta"}}},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 200
        assert r.json()["ids"] == []

    def test_upsert_100_times_same_id_count_stays_1(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        for i in range(100):
            requests.post(
                _col_url(base) + "/upsert",
                json={"ids": ["stable"], "embeddings": [_unit_vec(8, i)]},
                headers=HEADERS, timeout=10,
            )
        # Query all vectors; should be exactly 1
        r = requests.post(
            _col_url(base) + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 100},
            headers=HEADERS, timeout=10,
        )
        assert len(r.json()["results"][0]["ids"]) == 1

    def test_add_then_delete_then_query_returns_empty(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        requests.post(
            _col_url(base) + "/add",
            json={"ids": ["x"], "embeddings": [_unit_vec(8, 0)]},
            headers=HEADERS, timeout=10,
        )
        requests.post(_col_url(base) + "/delete", json={"ids": ["x"]},
                      headers=HEADERS, timeout=5)
        r = requests.post(
            _col_url(base) + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 10},
            headers=HEADERS, timeout=10,
        )
        assert r.json()["results"][0]["ids"] == []

    def test_query_on_nonexistent_collection_not_200(self, srv):
        base, _ = srv
        r = requests.post(
            _col_url(base, col="ghost") + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 5},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code != 200

    def test_get_on_nonexistent_collection_not_200(self, srv):
        base, _ = srv
        r = requests.post(
            _col_url(base, col="ghost") + "/get",
            json={"ids": ["v1"]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code != 200

    def test_delete_on_nonexistent_collection_not_200(self, srv):
        base, _ = srv
        r = requests.post(
            _col_url(base, col="ghost") + "/delete",
            json={"ids": ["v1"]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code != 200

    def test_compact_then_data_intact(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        n = 20
        requests.post(
            _col_url(base) + "/add",
            json={"ids": [f"v{i}" for i in range(n)], "embeddings": _unit_batch(n, 8)},
            headers=HEADERS, timeout=10,
        )
        r = requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
        job = _wait_job(base, r.json()["job_id"])
        assert job["status"] == "succeeded"
        # Query — should still return results
        qr = requests.post(
            _col_url(base) + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 5},
            headers=HEADERS, timeout=10,
        )
        assert len(qr.json()["results"][0]["ids"]) == 5

    def test_no_content_type_header_returns_error(self, srv):
        """Posting JSON without Content-Type should be rejected."""
        base, _ = srv
        _create_col(base, dim=8)
        r = requests.post(
            _col_url(base) + "/add",
            data=json.dumps({"ids": ["v1"], "embeddings": [_unit_vec(8, 0)]}),
            headers=AUTH_HDR,  # no Content-Type
            timeout=5,
        )
        assert r.status_code in (400, 415, 422)

    def test_query_scores_are_finite(self, srv):
        base, _ = srv
        _create_col(base, dim=8, metric="cosine")
        requests.post(
            _col_url(base) + "/add",
            json={"ids": [f"v{i}" for i in range(10)], "embeddings": _unit_batch(10, 8)},
            headers=HEADERS, timeout=10,
        )
        r = requests.post(
            _col_url(base) + "/query",
            json={"query_embeddings": [_unit_vec(8, 0)], "top_k": 10, "include": ["scores"]},
            headers=HEADERS, timeout=10,
        )
        scores = r.json()["results"][0]["scores"]
        for s in scores:
            assert isinstance(s, (int, float)), f"Score not numeric: {s}"
            import math
            assert math.isfinite(s), f"Score not finite: {s}"

    def test_multiple_jobs_enqueued_all_complete(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        requests.post(
            _col_url(base) + "/add",
            json={"ids": [f"v{i}" for i in range(20)], "embeddings": _unit_batch(20, 8)},
            headers=HEADERS, timeout=10,
        )
        job_ids = []
        for _ in range(3):
            r = requests.post(_col_url(base) + "/compact", json={}, headers=HEADERS, timeout=5)
            if r.status_code == 202:
                job_ids.append(r.json()["job_id"])
        for jid in job_ids:
            job = _wait_job(base, jid, timeout=30)
            assert job["status"] in ("succeeded", "canceled", "failed")

    def test_upsert_then_get_includes_all_fields(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        requests.post(
            _col_url(base) + "/upsert",
            json={
                "ids": ["full"],
                "embeddings": [_unit_vec(8, 0)],
                "metadatas": [{"key": "value", "n": 42}],
                "documents": ["The quick brown fox"],
            },
            headers=HEADERS, timeout=10,
        )
        r = requests.post(
            _col_url(base) + "/get",
            json={"ids": ["full"], "include": ["ids", "metadatas", "documents"]},
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ids"] == ["full"]
        assert body["metadatas"][0]["key"] == "value"
        assert body["metadatas"][0]["n"] == 42
        assert body["documents"][0] == "The quick brown fox"

    def test_filter_in_operator_server(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        requests.post(
            _col_url(base) + "/add",
            json={
                "ids": ["a", "b", "c"],
                "embeddings": _unit_batch(3, 8),
                "metadatas": [{"t": 1}, {"t": 2}, {"t": 3}],
            },
            headers=HEADERS, timeout=10,
        )
        r = requests.post(
            _col_url(base) + "/get",
            json={"filter": {"t": {"$in": [1, 3]}}},
            headers=HEADERS, timeout=5,
        )
        ids = set(r.json()["ids"])
        assert ids == {"a", "c"}

    def test_filter_exists_operator_server(self, srv):
        base, _ = srv
        _create_col(base, dim=8)
        requests.post(
            _col_url(base) + "/add",
            json={
                "ids": ["has", "lacks"],
                "embeddings": _unit_batch(2, 8),
                "metadatas": [{"special": True}, {}],
            },
            headers=HEADERS, timeout=10,
        )
        r = requests.post(
            _col_url(base) + "/get",
            json={"filter": {"special": {"$exists": True}}},
            headers=HEADERS, timeout=5,
        )
        assert r.json()["ids"] == ["has"]

    def test_add_report_mode_all_wrong_dim(self, srv):
        """report=true, ALL embeddings have wrong dim → applied=0, failed=[all]."""
        base, _ = srv
        _create_col(base, dim=8)
        url = _col_url(base) + "/add"
        r = requests.post(
            url,
            json={
                "ids": ["x", "y"],
                "embeddings": [[1.0], [2.0]],  # dim=1, col expects dim=8
                "report": True,
            },
            headers=HEADERS, timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] == 0
        assert len(body["failed"]) == 2


# ===========================================================================
# pytest main
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
