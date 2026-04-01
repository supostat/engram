#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { Lifecycle } from "./lifecycle.js";
import { TOOL_DEFINITIONS, executeTool } from "./tools.js";

function resolveSocketPath(): string {
  return process.env["ENGRAM_SOCKET_PATH"] ?? resolve(homedir(), ".engram", "engram.sock");
}

function resolveEngramBinary(): string {
  return process.env["ENGRAM_BIN"] ?? "engram";
}

async function main(): Promise<void> {
  const lifecycle = new Lifecycle({
    socketPath: resolveSocketPath(),
    engramBinary: resolveEngramBinary(),
  });

  const socketClient = await lifecycle.start();
  console.error("[engram-mcp] connected to engram-core");

  const server = new McpServer({
    name: "engram-mcp",
    version: "0.1.0",
  });

  for (const tool of TOOL_DEFINITIONS) {
    server.registerTool(
      tool.name,
      {
        description: tool.description,
        inputSchema: tool.schema,
      },
      async (params: Record<string, unknown>) => {
        try {
          const text = await executeTool(
            socketClient,
            tool.name,
            params,
          );
          return { content: [{ type: "text" as const, text }] };
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          return {
            content: [{ type: "text" as const, text: `Error: ${message}` }],
            isError: true,
          };
        }
      },
    );
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("[engram-mcp] MCP server listening on stdio");

  const handleShutdown = async (): Promise<void> => {
    console.error("[engram-mcp] shutting down");
    await lifecycle.shutdown();
    process.exit(0);
  };

  process.on("SIGINT", handleShutdown);
  process.on("SIGTERM", handleShutdown);
}

main().catch((error) => {
  console.error("[engram-mcp] fatal:", error);
  process.exit(1);
});
