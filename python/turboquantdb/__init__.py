"""
turboquantdb — high-performance embedded vector database.

Implements the TurboQuant algorithm (arXiv:2504.19874) for zero-training-time
vector quantization: 2–4 bits per coordinate, 8–16× less RAM than float32,
with provably unbiased inner-product estimation via QJL transforms.

Quick start::

    import numpy as np
    from turboquantdb import Database

    db = Database.open("mydb", dimension=1536, bits=4, metric="cosine")
    db.insert("doc1", np.random.randn(1536).astype(np.float32))
    results = db.search(np.random.randn(1536).astype(np.float32), top_k=5)

See ``Database.open`` for all parameters and ``Database.search`` for filter syntax.
"""
from .turboquantdb import Database, TurboQuantDB
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("turboquantdb")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Database", "TurboQuantDB", "__version__"]
