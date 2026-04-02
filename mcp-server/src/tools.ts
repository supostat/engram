import * as z from "zod/v4";
import type { SocketClient } from "./socket-client.js";
import { SocketError } from "./types.js";

export const TOOL_DEFINITIONS = [
  {
    name: "memory_store",
    description: "Store a new memory (decision, pattern, bugfix, context, antipattern)",
    schema: z.object({
      context: z.string().describe("Situation/context where the action occurred"),
      action: z.string().describe("Action or decision taken"),
      result: z.string().describe("Result or outcome"),
      memory_type: z
        .enum(["decision", "pattern", "bugfix", "context", "antipattern"])
        .default("decision")
        .describe(
          "decision: architecture/design choice with reasoning. " +
          "pattern: reusable solution applicable to future tasks. " +
          "bugfix: diagnosis and fix for a specific bug. " +
          "context: project fact, setup info, or phase completion status. " +
          "antipattern: what NOT to do, with reasoning why."
        ),
      tags: z.string().optional().describe("Comma-separated tags"),
      project: z.string().optional().describe("Project identifier"),
    }),
  },
  {
    name: "memory_search",
    description: "Search for relevant memories using hybrid vector + sparse search",
    schema: z.object({
      query: z.string().describe("Search query"),
      limit: z.number().default(10).describe("Maximum results"),
      project: z.string().optional().describe("Filter by project"),
    }),
  },
  {
    name: "memory_judge",
    description: "Rate a memory's quality (feeds Q-Learning router)",
    schema: z.object({
      memory_id: z.string().describe("Memory ID to judge"),
      query: z.string().optional().describe("Query context for relevance"),
      score: z.number().min(0).max(1).optional().describe("Explicit score 0.0-1.0"),
    }),
  },
  {
    name: "memory_status",
    description: "Get system status: memory count, index size, pending judgments",
    schema: z.object({}),
  },
  {
    name: "memory_config",
    description: "Read current configuration (read-only in this version)",
    schema: z.object({
      action: z.enum(["get", "set"]).default("get"),
    }),
  },
  {
    name: "memory_export",
    description: "Export all active memories as portable JSON",
    schema: z.object({}),
  },
  {
    name: "memory_import",
    description: "Import memories from exported JSON (merge mode, skips duplicates)",
    schema: z.object({
      version: z.number().describe("Export format version (must be 1)"),
      memories: z
        .array(
          z.object({
            id: z.string(),
            memory_type: z.string(),
            context: z.string(),
            action: z.string(),
            result: z.string(),
            score: z.number(),
            tags: z.string().nullable().optional(),
            project: z.string().nullable().optional(),
            parent_id: z.string().nullable().optional(),
            source_ids: z.string().nullable().optional(),
            insight_type: z.string().nullable().optional(),
            created_at: z.string(),
            updated_at: z.string(),
            used_count: z.number(),
            last_used_at: z.string().nullable().optional(),
          }),
        )
        .describe("Array of memory objects"),
    }),
  },
  {
    name: "memory_consolidate_preview",
    description: "Find deduplication candidates without applying changes",
    schema: z.object({
      stale_days: z.number().optional().describe("Age threshold in days"),
      min_score: z.number().optional().describe("Minimum score threshold"),
    }),
  },
  {
    name: "memory_consolidate",
    description: "Analyze consolidation opportunities with LLM",
    schema: z.object({
      stale_days: z.number().optional(),
      min_score: z.number().optional(),
    }),
  },
  {
    name: "memory_consolidate_apply",
    description: "Apply consolidation recommendations (merge/delete/archive)",
    schema: z.object({
      stale_days: z.number().optional(),
      min_score: z.number().optional(),
    }),
  },
  {
    name: "memory_insights",
    description: "List, generate, or delete derived knowledge insights",
    schema: z.object({
      action: z.enum(["list", "generate", "delete"]).default("list"),
      id: z.string().optional().describe("Insight ID (for delete action)"),
    }),
  },
] as const;

export async function executeTool(
  client: SocketClient,
  name: string,
  params: Record<string, unknown>,
): Promise<string> {
  try {
    const result = await client.call(name, params);
    return JSON.stringify(result, null, 2);
  } catch (error) {
    if (error instanceof SocketError) {
      throw new Error(`[${error.code ?? "?"}] ${error.message}`);
    }
    throw error;
  }
}
