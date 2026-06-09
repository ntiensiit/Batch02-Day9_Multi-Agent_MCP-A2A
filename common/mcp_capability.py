"""Educational MCP-compatible simulation module.

Simulates Model Context Protocol (MCP) discovery (list tools) and invocation (call tool) semantics.
"""

from typing import Any, Callable, Dict, List, TypedDict

class MCPToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPServer:
    """Simulates an MCP server managing a set of tools."""

    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, MCPToolDefinition] = {}

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], func: Callable) -> None:
        """Register a tool with the MCP server."""
        self.tools[name] = func
        self.tool_definitions[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }

    def list_tools(self) -> List[MCPToolDefinition]:
        """List all tools exposed by the MCP server (Discovery)."""
        return list(self.tool_definitions.values())

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Invoke a tool by name with arguments (Invocation)."""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found on MCP server {self.name}")
        try:
            return str(self.tools[name](**arguments))
        except Exception as e:
            return f"Error executing tool {name}: {str(e)}"
