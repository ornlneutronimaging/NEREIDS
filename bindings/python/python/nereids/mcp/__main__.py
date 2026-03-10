"""Run the NEREIDS MCP server: python -m nereids.mcp"""

from nereids.mcp import _check_fastmcp

_check_fastmcp()

from nereids.mcp.server import mcp  # noqa: E402

mcp.run()
