"""
Launcher shim for the tqdb-server binary.

When installed from a wheel, the server binary lives alongside the package in
``_bin/``.  When running from a ``maturin develop`` checkout, it falls back to
the local cargo build artefact so developers don't need to copy the file.
"""
import os
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    bin_name = "tqdb-server.exe" if sys.platform == "win32" else "tqdb-server"
    candidates = [
        os.path.join(here, "_bin", bin_name),                          # installed wheel
        os.path.join(here, "..", "..", "server", "target",             # maturin develop
                     "release", bin_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            os.execv(path, [path] + sys.argv[1:])

    import platform
    arch = platform.machine().lower()
    extra = ""
    if arch in ("aarch64", "arm64") and sys.platform.startswith("linux"):
        extra = (
            "\n\nNote: tqdb-server is not bundled for Linux aarch64 wheels.\n"
            "Build it from source with:\n  cd server && cargo build --release"
        )
    print(
        f"tqdb-server binary not found in expected locations:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\n\nBuild it with:\n  cd server && cargo build --release"
        + extra,
        file=sys.stderr,
    )
    sys.exit(1)
