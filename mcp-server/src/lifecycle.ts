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
    await this.verifyApiKeys(client);
    this.client = client;
    this.state = LifecycleState.Connected;
    console.error("[engram-mcp] reconnected to existing engram-core");
    this.startHealthCheck();
    return client;
  }

  private async verifyApiKeys(client: SocketClient): Promise<void> {
    const config = (await client.call("memory_config", {
      action: "get",
    })) as Record<string, Record<string, unknown>>;
    const embedding = config?.embedding;
    if (embedding && embedding.provider !== "deterministic" && !embedding.has_api_key) {
      console.error(
        "[engram-mcp] orphan core missing embedding api key, killing and respawning",
      );
      await client.close();
      this.killOrphan();
      throw new Error("orphan core missing embedding api key");
    }
  }

  private killOrphan(): void {
    try {
      unlinkSync(this.socketPath);
    } catch {
      /* already gone */
    }
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
        this.attemptReconnect();
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
    this.healthTimer = setInterval(() => {
      this.performHealthCheck();
    }, this.healthIntervalMs);
  }

  private performHealthCheck(): void {
    if (this.state !== LifecycleState.Connected || !this.client) return;

    this.client.call("memory_status", {}).catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error("[engram-mcp] health check failed:", message);
      this.attemptReconnect();
    });
  }

  private attemptReconnect(): void {
    if (this.state === LifecycleState.Reconnecting) return;
    this.state = LifecycleState.Reconnecting;

    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }

    if (this.client) {
      this.client.close().catch(() => {});
      this.client = null;
    }

    this.reconnectWithBackoff().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error("[engram-mcp] reconnection exhausted:", message);
      this.state = LifecycleState.Dead;
    });
  }

  private async reconnectWithBackoff(): Promise<void> {
    const backoffs = [1000, 2000, 4000];
    for (let attempt = 0; attempt < backoffs.length; attempt++) {
      await sleep(backoffs[attempt]);
      console.error(`[engram-mcp] reconnect attempt ${attempt + 1}/${backoffs.length}`);
      try {
        await this.start();
        return;
      } catch {
        // next attempt
      }
    }
    throw new Error("reconnect attempts exhausted");
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
