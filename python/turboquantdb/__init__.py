from .turboquantdb import Database, TurboQuantDB
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("turboquantdb")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Database", "TurboQuantDB", "__version__"]
