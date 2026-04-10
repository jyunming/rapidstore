from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tqdb import Database


def _server_exe() -> Path:
    base = Path(__file__).resolve().parents[1] / "server" / "target" / "release"
    if os.name == "nt":
        return base / "tqdb-server.exe"
    return base / "tqdb-server"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_json(method: str, url: str, body=None, headers=None, timeout: float = 6):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(
        url, data=data, headers=request_headers, method=method.upper()
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


@pytest.mark.slow
def test_embedded_mixed_workload_reopen_consistency(tmp_path):
    rng = np.random.default_rng(123)
    prng = random.Random(123)
    path = str(tmp_path / "db")
    d = 32
    db = Database.open(path, dimension=d, bits=4, metric="ip")
    shadow: dict[str, dict] = {}

    for i in range(1200):
        op = prng.choice(["insert", "upsert", "delete", "update", "get", "search"])
        idx = prng.randint(0, 250)
        id_ = f"id_{idx}"
        vec = rng.standard_normal(d).astype(np.float32)
        meta = {"bucket": idx % 7, "step": i}
        doc = f"doc-{i}"

        if op == "insert":
            try:
                db.insert(id_, vec, metadata=meta, document=doc)
                shadow[id_] = {"metadata": meta, "document": doc}
            except Exception:
                # Duplicate insert is acceptable; shadow remains unchanged.
                pass
        elif op == "upsert":
            db.upsert(id_, vec, metadata=meta, document=doc)
            shadow[id_] = {"metadata": meta, "document": doc}
        elif op == "delete":
            deleted = db.delete(id_)
            if deleted:
                shadow.pop(id_, None)
        elif op == "update":
            try:
                db.update(id_, vec, metadata=meta, document=doc)
                if id_ in shadow:
                    shadow[id_] = {"metadata": meta, "document": doc}
            except Exception:
                # Missing-ID update is acceptable.
                pass
        elif op == "get":
            got = db.get(id_)
            if id_ in shadow:
                assert got is not None
            else:
                assert got is None
        else:
            out = db.search(vec, top_k=10)
            assert len(out) <= 10

        if i % 200 == 0:
            db.close()
            db = Database.open(path, dimension=d)
            assert db.count() == len(shadow)
            assert set(db.list_all()) == set(shadow.keys())

    db.close()
    reopened = Database.open(path, dimension=d)
    assert reopened.count() == len(shadow)
    assert set(reopened.list_all()) == set(shadow.keys())


@pytest.mark.slow
def test_embedded_reinsert_churn_many_ids_survives_restart(tmp_path):
    path = str(tmp_path / "db")
    d = 16
    db = Database.open(path, dimension=d, bits=4, metric="ip")
    v = np.arange(d, dtype=np.float32)

    ids = [f"id_{i}" for i in range(50)]
    for round_idx in range(12):
        for id_ in ids:
            db.upsert(id_, v, metadata={"round": round_idx}, document=f"r{round_idx}")
            assert db.get(id_) is not None
            db.delete(id_)
            db.upsert(id_, v, metadata={"round": round_idx}, document=f"r{round_idx}")

        db.close()
        db = Database.open(path, dimension=d)
        # After each restart, every churned ID should still exist.
        for id_ in ids:
            got = db.get(id_)
            assert got is not None, f"missing after restart: {id_} round={round_idx}"

    assert db.count() == len(ids)


@pytest.mark.slow
def test_server_mixed_concurrent_soak_no_timeouts_or_500():
    exe = _server_exe()
    if not exe.exists():
        pytest.skip(f"server binary missing: {exe}")

    with tempfile.TemporaryDirectory() as td:
        local_root = Path(td) / "data"
        local_root.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        env = os.environ.copy()
        env["TQ_LOCAL_ROOT"] = str(local_root)
        env["TQ_SERVER_ADDR"] = f"127.0.0.1:{port}"
        proc = subprocess.Popen(
            [str(exe)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            ready = False
            for _ in range(80):
                st, _ = _http_json("GET", f"http://127.0.0.1:{port}/healthz")
                if st == 200:
                    ready = True
                    break
                time.sleep(0.2)
            assert ready

            headers = {"Authorization": "ApiKey dev-key"}
            st, body = _http_json(
                "POST",
                f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections",
                {"name": "soak", "dimension": 8, "bits": 4, "metric": "ip"},
                headers,
            )
            assert st == 201, body

            statuses: list[int | str] = []
            lock = threading.Lock()
            stop_at = time.time() + 6.0

            def _worker(seed: int):
                rnd = random.Random(seed)
                while time.time() < stop_at:
                    op = rnd.choice(["add", "get", "query", "delete"])
                    i = rnd.randint(0, 200)
                    if op == "add":
                        payload = {"ids": [f"id_{i}"], "embeddings": [[1.0] * 8]}
                    elif op == "get":
                        payload = {"ids": [f"id_{i}"]}
                    elif op == "query":
                        payload = {"query_embeddings": [[0.0] * 8], "top_k": 3}
                    else:
                        payload = {"ids": [f"id_{i}"]}

                    endpoint = op
                    try:
                        st_i, _ = _http_json(
                            "POST",
                            f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/soak/{endpoint}",
                            payload,
                            headers,
                            timeout=2.5,
                        )
                    except TimeoutError:
                        st_i = "timeout"
                    with lock:
                        statuses.append(st_i)

            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert "timeout" not in statuses, statuses[:20]
            assert 500 not in statuses, statuses[:20]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.mark.slow
def test_server_restart_roundtrip_write_then_read():
    exe = _server_exe()
    if not exe.exists():
        pytest.skip(f"server binary missing: {exe}")

    with tempfile.TemporaryDirectory() as td:
        local_root = Path(td) / "data"
        local_root.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        env = os.environ.copy()
        env["TQ_LOCAL_ROOT"] = str(local_root)
        env["TQ_SERVER_ADDR"] = f"127.0.0.1:{port}"
        headers = {"Authorization": "ApiKey dev-key"}

        proc = subprocess.Popen(
            [str(exe)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            ready = False
            for _ in range(80):
                st, _ = _http_json("GET", f"http://127.0.0.1:{port}/healthz")
                if st == 200:
                    ready = True
                    break
                time.sleep(0.2)
            assert ready

            st, body = _http_json(
                "POST",
                f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections",
                {"name": "durable", "dimension": 8, "bits": 4, "metric": "ip"},
                headers,
            )
            assert st == 201, body

            st, body = _http_json(
                "POST",
                f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/durable/add",
                {"ids": ["doc1"], "embeddings": [[1.0] * 8]},
                headers,
            )
            assert st in (200, 201), body
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Restart server on same storage root and verify roundtrip durability.
        port2 = _find_free_port()
        env2 = os.environ.copy()
        env2["TQ_LOCAL_ROOT"] = str(local_root)
        env2["TQ_SERVER_ADDR"] = f"127.0.0.1:{port2}"
        proc2 = subprocess.Popen(
            [str(exe)], env=env2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            ready = False
            for _ in range(80):
                st, _ = _http_json("GET", f"http://127.0.0.1:{port2}/healthz")
                if st == 200:
                    ready = True
                    break
                time.sleep(0.2)
            assert ready

            st, body = _http_json(
                "POST",
                f"http://127.0.0.1:{port2}/v1/tenants/dev/databases/db/collections/durable/get",
                {"ids": ["doc1"]},
                headers,
            )
            assert st == 200, body
            payload = json.loads(body)
            assert payload.get("ids") == ["doc1"], payload
        finally:
            proc2.terminate()
            try:
                proc2.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc2.kill()


@pytest.mark.slow
def test_server_concurrent_distinct_collection_create_has_no_500_and_no_loss():
    exe = _server_exe()
    if not exe.exists():
        pytest.skip(f"server binary missing: {exe}")

    with tempfile.TemporaryDirectory() as td:
        local_root = Path(td) / "data"
        local_root.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        env = os.environ.copy()
        env["TQ_LOCAL_ROOT"] = str(local_root)
        env["TQ_SERVER_ADDR"] = f"127.0.0.1:{port}"
        headers = {"Authorization": "ApiKey dev-key"}

        proc = subprocess.Popen(
            [str(exe)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            ready = False
            for _ in range(80):
                st, _ = _http_json("GET", f"http://127.0.0.1:{port}/healthz")
                if st == 200:
                    ready = True
                    break
                time.sleep(0.2)
            assert ready

            statuses: list[int] = []
            lock = threading.Lock()

            def _worker(i: int):
                st, _ = _http_json(
                    "POST",
                    f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections",
                    {"name": f"sim_{i}", "dimension": 8, "bits": 4, "metric": "ip"},
                    headers,
                )
                with lock:
                    statuses.append(st)

            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert 500 not in statuses, statuses
            assert all(s == 201 for s in statuses), statuses

            st, body = _http_json(
                "GET",
                f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections",
                None,
                headers,
            )
            assert st == 200, body
            payload = json.loads(body)
            names = {c["name"] for c in payload.get("collections", [])}
            expected = {f"sim_{i}" for i in range(50)}
            assert expected <= names, (len(expected - names), sorted(list(expected - names))[:5])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
