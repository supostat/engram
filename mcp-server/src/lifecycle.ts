import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, unlinkSync } from "node:fs";
import { SocketClient } from "./socket-client.js";
import { LifecycleState, type LifecycleConfig } from "./types.js";

export class Lifecycle {
  private state = LifecycleState.Disconnected;
  private process: ChildProcess | null = null;
  private client: SocketClient | null = null;
  private healthTimer: ReturnType<typeof setInterval> | null = null;
  private readonly socketPath: string;
  private readonly engramBinary: string;
  private readonly spawnTimeoutMs: number;
  private readonly healthIntervalMs: number;

  constructor(config: LifecycleConfig) {
    this.socketPath = config.socketPath;
    this.engramBinary = config.engramBinary;
    this.spawnTimeoutMs = config.spawnTimeoutMs ?? 30000;
    this.healthIntervalMs = config.healthIntervalMs ?? 60000;
  }

  async start(): Promise<SocketClient> {
    if (existsSync(this.socketPath)) {
      try {
        return await this.connectToExisting();
      } catch {
        console.error("[engram-mcp] stale socket, removing");
        try { unlinkSync(this.socketPath); } catch { /* already gone */ }
      }
    }

    this.state = LifecycleState.Spawning;
    this.spawnProcess();

    this.state = LifecycleState.WaitingForSocket;
    await this.waitForSocket();

    this.client = new SocketClient({ socketPath: this.socketPath });
    await this.client.connect();
    this.state = LifecycleState.Connected;

    this.startHealthCheck();
    return this.client;
  }

  async shutdown(): Promise<void> {
    if (this.healthTimer) clearInterval(this.healthTimer);

    if (this.client) {
      try { await this.client.close(); } catch { /* best effort */ }
    }

    if (this.process && !this.process.killed) {
      this.process.kill("SIGTERM");
      await sleep(1000);
      if (!this.process.killed) {
        this.process.kill("SIGKILL");
      }
    }

    this.state = LifecycleState.Disconnected;
  }

  getState(): LifecycleState {
    return this.state;
  }

  private async connectToExisting(): Promise<SocketClient> {
    const client = new SocketClient({ socketPath: this.socketPath });
    await client.connect();
    await client.call("memory_status", {});
    this.client = client;
    this.state = LifecycleState.Connected;
    console.error("[engram-mcp] reconnected to existing engram-core");
    this.startHealthCheck();
    return client;
  }

  private spawnProcess(): void {
    this.process = spawn(this.engramBinary, ["server"], {
      stdio: ["ignore", "pipe", "pipe"],
      detached: false,
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      console.error(`[engram-core] ${data.toString().trimEnd()}`);
    });

    this.process.on("exit", (code) => {
      console.error(`[engram-mcp] engram-core exited with code ${code}`);
      if (this.state === LifecycleState.Connected) {
        this.state = LifecycleState.Dead;
      }
    });
  }

  private async waitForSocket(): Promise<void> {
    const deadline = Date.now() + this.spawnTimeoutMs;
    while (Date.now() < deadline) {
      if (existsSync(this.socketPath)) return;
      await sleep(100);
    }
    throw new Error(`socket not created after ${this.spawnTimeoutMs}ms`);
  }

  private startHealthCheck(): void {
    this.healthTimer = setInterval(async () => {
      try {
        await this.client!.call("memory_status", {});
      } catch (error) {
        console.error("[engram-mcp] health check failed:", error);
        this.state = LifecycleState.Reconnecting;
      }
    }, this.healthIntervalMs);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
