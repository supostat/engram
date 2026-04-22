import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveProjectDir } from "./index.js";

const ENV_KEY = "ENGRAM_PROJECT_DIR";

describe("resolveProjectDir", () => {
  let tempRoot: string;
  let originalEnv: string | undefined;

  beforeEach(() => {
    tempRoot = mkdtempSync(join(tmpdir(), "engram-project-dir-"));
    originalEnv = process.env[ENV_KEY];
    delete process.env[ENV_KEY];
  });

  afterEach(() => {
    rmSync(tempRoot, { recursive: true, force: true });
    if (originalEnv === undefined) {
      delete process.env[ENV_KEY];
    } else {
      process.env[ENV_KEY] = originalEnv;
    }
  });

  it("finds the .engram marker in the start directory", () => {
    mkdirSync(join(tempRoot, ".engram"));
    expect(resolveProjectDir(tempRoot)).toBe(tempRoot);
  });

  it("walks up the directory tree to find the marker", () => {
    mkdirSync(join(tempRoot, ".engram"));
    const nested = join(tempRoot, "a", "b", "c");
    mkdirSync(nested, { recursive: true });
    expect(resolveProjectDir(nested)).toBe(tempRoot);
  });

  it("respects the ENGRAM_PROJECT_DIR env override over walk-up", () => {
    // Override target: must exist.
    const envTarget = mkdtempSync(join(tmpdir(), "engram-env-target-"));
    try {
      process.env[ENV_KEY] = envTarget;
      // Start directory lives elsewhere and has no marker — walk-up would fail.
      expect(resolveProjectDir(tempRoot)).toBe(envTarget);
    } finally {
      rmSync(envTarget, { recursive: true, force: true });
    }
  });

  it("throws when no .engram marker is found in any ancestor", () => {
    // tempRoot lives under tmpdir(); none of its ancestors carry a .engram marker.
    expect(() => resolveProjectDir(tempRoot)).toThrow(/no \.engram\//);
  });
});
