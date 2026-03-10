"""NEREIDS MCP server -- expose nuclear data tools to AI agents."""


def _check_fastmcp():
    """Raise a helpful error if fastmcp is not installed."""
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        raise ImportError(
            "fastmcp is required for the MCP server. "
            "Install it with: pip install nereids[mcp]"
        ) from None


# Lazy import: only create the server when actually accessed.
def __getattr__(name):
    if name == "mcp":
        _check_fastmcp()
        from nereids.mcp.server import mcp

        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["mcp"]
