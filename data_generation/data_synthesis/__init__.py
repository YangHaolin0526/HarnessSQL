"""Harness-aware task synthesis for the local Spider 2.0 SQLite databases."""

__all__ = ["CATALOG_VERSION", "CatalogBuilder"]


def __getattr__(name: str):
    if name in __all__:
        from .catalog import CATALOG_VERSION, CatalogBuilder

        return {"CATALOG_VERSION": CATALOG_VERSION, "CatalogBuilder": CatalogBuilder}[name]
    raise AttributeError(name)
