import json
import os
import socket
import subprocess
import tempfile
import time
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def _server_exe() -> Path:
    base = Path(__file__).resolve().parents[1] / "server" / "target" / "release"
    if os.name == "nt":
        return base / "tqdb-server.exe"
    return base / "tqdb-server"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_json(method: str, url: str, body=None, headers=None, timeout: float = 10):
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


@pytest.fixture
def running_server():
    exe = _server_exe()
    if not exe.exists():
        pytest.skip(f"server binary missing: {exe}")

    with tempfile.TemporaryDirectory() as td:
        local_root = str(Path(td) / "data")
        port = _find_free_port()
        env = os.environ.copy()
        env["TQ_LOCAL_ROOT"] = local_root
        env["TQ_SERVER_ADDR"] = f"127.0.0.1:{port}"

        proc = subprocess.Popen(
            [str(exe)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            ready = False
            for _ in range(80):
                status, _ = _http_json("GET", f"http://127.0.0.1:{port}/healthz")
                if status == 200:
                    ready = True
                    break
                time.sleep(0.2)
            if not ready:
                pytest.fail("server failed to become healthy")
            yield {"port": port, "local_root": local_root}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _auth_headers():
    return {"Authorization": "ApiKey dev-key"}


def _create_collection(port: int, name: str = "c1"):
    return _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections",
        {"name": name, "dimension": 8, "bits": 4, "metric": "ip"},
        _auth_headers(),
    )


def test_server_create_then_add_succeeds(running_server):
    port = running_server["port"]
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    status, body = _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/add",
        {"ids": ["doc1"], "embeddings": [[1.0] * 8]},
        _auth_headers(),
    )
    # A healthy data plane should not internal-error immediately after create.
    assert status in (200, 201), body


def test_server_create_then_get_query_do_not_internal_error(running_server):
    port = running_server["port"]
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    get_status, get_body = _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/get",
        {"ids": ["missing"]},
        _auth_headers(),
    )
    query_status, query_body = _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/query",
        {"query_embeddings": [[0.0] * 8], "top_k": 1},
        _auth_headers(),
    )

    assert get_status != 500, get_body
    assert query_status != 500, query_body


def test_server_rejects_collection_path_traversal(running_server):
    port = running_server["port"]
    status, _ = _create_collection(port, "..\\..\\escaped")
    # Collection names should be treated as simple identifiers, not paths.
    assert status in (400, 403), f"unexpected status for traversal name: {status}"


def test_server_create_collection_does_not_pollute_root_manifest(running_server):
    port = running_server["port"]
    local_root = Path(running_server["local_root"])
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    # Root-level DB artifacts indicate path-resolution bugs between scoped and root layouts.
    assert not (local_root / "manifest.json").exists()
    assert not (local_root / "quantizer.bin").exists()
    assert not (local_root / "live_ids.bin").exists()


def test_server_rejects_tenant_database_path_traversal_with_admin_key():
    exe = _server_exe()
    if not exe.exists():
        pytest.skip(f"server binary missing: {exe}")

    with tempfile.TemporaryDirectory() as td:
        local_root = Path(td) / "data"
        local_root.mkdir(parents=True, exist_ok=True)
        auth_path = Path(td) / "auth_store.json"
        auth_path.write_text(
            json.dumps(
                {
                    "api_keys": [{"key": "root-key", "subject": "root"}],
                    "principals": [
                        {
                            "subject": "root",
                            "tenant_id": None,
                            "roles": ["admin"],
                            "scopes": ["read", "write", "admin"],
                        }
                    ],
                    "role_bindings": [
                        {
                            "subject": "root",
                            "tenant": None,
                            "database": None,
                            "collection": None,
                            "actions": ["read", "write", "admin"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        port = _find_free_port()
        env = os.environ.copy()
        env["TQ_LOCAL_ROOT"] = str(local_root)
        env["TQ_SERVER_ADDR"] = f"127.0.0.1:{port}"
        env["TQ_AUTH_STORE_PATH"] = str(auth_path)

        proc = subprocess.Popen(
            [str(exe)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            ready = False
            for _ in range(80):
                status, _ = _http_json("GET", f"http://127.0.0.1:{port}/healthz")
                if status == 200:
                    ready = True
                    break
                time.sleep(0.2)
            assert ready

            headers = {"Authorization": "ApiKey root-key"}
            status, body = _http_json(
                "POST",
                f"http://127.0.0.1:{port}/v1/tenants/..%5C..%5Cescaped/databases/..%5C..%5Cdb/collections",
                {"name": "c1", "dimension": 8, "bits": 4, "metric": "ip"},
                headers,
            )
            assert status in (400, 403), body
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_server_async_job_endpoints_return_promptly(running_server):
    port = running_server["port"]
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    endpoints = [
        ("index", {}),
        ("compact", {}),
        ("snapshot", {}),
        ("restore", {"snapshot_name": "missing-snapshot"}),
    ]

    for endpoint, payload in endpoints:
        try:
            status, body = _http_json(
                "POST",
                f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/{endpoint}",
                payload,
                _auth_headers(),
                timeout=4,
            )
        except TimeoutError:
            pytest.fail(f"{endpoint} endpoint timed out instead of returning a response")

        # Enqueue APIs should return quickly with either Accepted or a structured error.
        assert status in (202, 400, 404, 409, 429), (endpoint, status, body)


def test_server_add_bad_dimension_is_client_error_not_internal(running_server):
    port = running_server["port"]
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    status, body = _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/add",
        {"ids": ["doc1"], "embeddings": [[1.0] * 7]},
        _auth_headers(),
    )
    assert status in (400, 422), body


def test_server_query_bad_dimension_is_client_error_not_internal(running_server):
    port = running_server["port"]
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    status, body = _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/query",
        {"query_embeddings": [[0.0] * 7], "top_k": 1},
        _auth_headers(),
    )
    assert status in (400, 422), body


def test_server_delete_missing_id_is_not_internal_error(running_server):
    port = running_server["port"]
    status, body = _create_collection(port, "c1")
    assert status == 201, body

    status, body = _http_json(
        "POST",
        f"http://127.0.0.1:{port}/v1/tenants/dev/databases/db/collections/c1/delete",
        {"ids": ["missing-id"]},
        _auth_headers(),
    )
    # Deleting a missing ID should return a normal API response, never 500.
    assert status in (200, 204, 404), body


def test_server_concurrent_create_same_collection_is_linearizable(running_server):
    port = running_server["port"]
    statuses = []
    errors = []

    def _worker():
        try:
            st, _ = _create_collection(port, "race_col")
            statuses.append(st)
        except BaseException as e:  # pragma: no cover - safety for thread exceptions
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # Exactly one creator should succeed, all others should see conflict.
    assert statuses.count(201) == 1, statuses
    assert all(s in (201, 409) for s in statuses), statuses
