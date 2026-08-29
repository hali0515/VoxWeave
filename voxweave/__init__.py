import importlib.metadata

# pyproject.toml is the single source of truth for the version; a hardcoded
# string here drifts silently across releases (0.14.0 shipped reporting 0.1.0).
try:
    __version__ = importlib.metadata.version("voxweave")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
