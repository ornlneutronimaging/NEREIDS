"""NEREIDS MCP server -- expose nuclear data tools to AI agents."""


def _fastmcp_available() -> bool:
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        return False
    return True


_FASTMCP_INSTALL_MSG = (
    "fastmcp is required for the MCP server. "
    "Install it with: pip install nereids[mcp]"
)


# Lazy import: only create the server when actually accessed.
# Raises AttributeError (not ImportError) when fastmcp is missing so that
# attribute-walking tools (pdoc, IDE introspection) treat `mcp` as absent
# rather than failing with an uncaught ImportError. The install instruction
# is preserved in the AttributeError message.
def __getattr__(name):
    if name == "mcp":
        if not _fastmcp_available():
            raise AttributeError(
                f"module {__name__!r} attribute 'mcp' is unavailable: "
                f"{_FASTMCP_INSTALL_MSG}"
            )
        from nereids.mcp.server import mcp

        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main():
    """Run the NEREIDS MCP server over stdio."""
    if not _fastmcp_available():
        raise ImportError(_FASTMCP_INSTALL_MSG)
    from nereids.mcp.server import mcp

    mcp.run()


__all__ = ["mcp", "main"]
