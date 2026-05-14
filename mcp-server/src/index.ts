#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { existsSync, statSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";
import { Lifecycle } from "./lifecycle.js";
import { TOOL_DEFINITIONS, executeTool } from "./tools.js";

const PROJECT_DIR_MARKER = ".engram";
const PROJECT_SOCKET_RELATIVE = join(PROJECT_DIR_MARKER, "engram.sock");

class ProjectDirNotFoundError extends Error {
  constructor(startDir: string) {
    super(`no ${PROJECT_DIR_MARKER}/ found in ${startDir} or any ancestor`);
    this.name = "ProjectDirNotFoundError";
  }
}

function isExistingDirectory(candidate: string): boolean {
  try {
    return statSync(candidate).isDirectory();
  } catch {
    return false;
  }
}

export function resolveProjectDir(start: string = process.cwd()): string {
  const envOverride = process.env["ENGRAM_PROJECT_DIR"];
  if (envOverride && isAbsolute(envOverride) && isExistingDirectory(envOverride)) {
    return envOverride;
  }
  let current = resolve(start);
  while (true) {
    const marker = join(current, PROJECT_DIR_MARKER);
    if (isExistingDirectory(marker)) {
      return current;
    }
    const parent = dirname(current);
    if (parent === current) {
      throw new ProjectDirNotFoundError(start);
    }
    current = parent;
  }
}

function resolveSocketPath(): string {
  const envSocket = process.env["ENGRAM_SOCKET_PATH"];
  if (envSocket) {
    return envSocket;
  }
  try {
    const projectDir = resolveProjectDir();
    return join(projectDir, PROJECT_SOCKET_RELATIVE);
  } catch (error) {
    if (!(error instanceof ProjectDirNotFoundError)) {
      throw error;
    }
    return resolve(homedir(), PROJECT_DIR_MARKER, "engram.sock");
  }
}

function resolveEngramBinary(): string {
  return process.env["ENGRAM_BIN"] ?? "engram";
}

async function main(): Promise<void> {
  const socketPath = resolveSocketPath();
  if (!existsSync(socketPath)) {
    // Log only; lifecycle will spawn the daemon which creates the socket.
    console.error(`[engram-mcp] socket not present yet at ${socketPath}`);
  }
  const lifecycle = new Lifecycle({
    socketPath,
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

// Only run main when this file is the entry point (node index.js). Importing
// `resolveProjectDir` (or any other export) from tests must NOT trigger
// Lifecycle.start, which spawns the `engram` binary and crashes with ENOENT
// in CI environments that don't have it on PATH.
const invokedAsEntryPoint =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (invokedAsEntryPoint) {
  main().catch((error) => {
    console.error("[engram-mcp] fatal:", error);
    process.exit(1);
  });
}
