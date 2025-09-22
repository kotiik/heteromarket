# Re-export the public API at the package root
from .core import optimize_portfolio, find_equilibrium_prices

# Optional: package version from installed metadata
try:
    from importlib.metadata import version, PackageNotFoundError  # py3.8+
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

try:
    __version__ = version("heteromarket") if version else "0.0.0"
except PackageNotFoundError:  # during dev / editable installs
    __version__ = "0.0.0"

# Public surface
__all__ = [
    "optimize_portfolio",
    "find_equilibrium_prices",
    "__version__",
]
