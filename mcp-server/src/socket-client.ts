import { createConnection, type Socket } from "node:net";
import { randomUUID } from "node:crypto";
import type { SocketRequest, SocketResponse, SocketClientConfig } from "./types.js";
import { SocketError, ConnectionLostError } from "./types.js";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class SocketClient {
  private socket: Socket | null = null;
  private buffer = "";
  private pending = new Map<string, PendingRequest>();
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 3;
  private readonly reconnectBackoffMs = [1000, 2000, 4000];
  private readonly socketPath: string;
  private readonly connectTimeoutMs: number;
  private readonly requestTimeoutMs: number;

  constructor(config: SocketClientConfig) {
    this.socketPath = config.socketPath;
    this.connectTimeoutMs = config.connectTimeoutMs ?? 5000;
    this.requestTimeoutMs = config.requestTimeoutMs ?? 10000;
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = createConnection(this.socketPath);

      const connectTimer = setTimeout(() => {
        socket.destroy();
        reject(new SocketError(`connect timeout after ${this.connectTimeoutMs}ms`));
      }, this.connectTimeoutMs);

      socket.on("connect", () => {
        clearTimeout(connectTimer);
        this.socket = socket;
        this.reconnectAttempts = 0;
        resolve();
      });

      socket.on("error", (error) => {
        clearTimeout(connectTimer);
        reject(new SocketError(error.message));
      });

      socket.on("data", (chunk: Buffer) => this.handleData(chunk));

      socket.on("close", () => {
        this.socket = null;
        this.rejectAllPending("connection closed");
      });
    });
  }

  async call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    if (!this.socket?.writable) {
      await this.reconnect();
    }

    const id = randomUUID();
    const request: SocketRequest = { id, method, params };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new SocketError(`request timeout: ${method} (${this.requestTimeoutMs}ms)`));
      }, this.requestTimeoutMs);

      this.pending.set(id, { resolve, reject, timer });

      this.socket!.write(JSON.stringify(request) + "\n", "utf8", (error) => {
        if (error) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(new SocketError(error.message));
        }
      });
    });
  }

  async close(): Promise<void> {
    this.rejectAllPending("client closing");
    return new Promise((resolve) => {
      if (!this.socket) {
        resolve();
        return;
      }
      this.socket.end(() => {
        this.socket = null;
        resolve();
      });
    });
  }

  private async reconnect(): Promise<void> {
    while (this.reconnectAttempts < this.maxReconnectAttempts) {
      const backoff = this.reconnectBackoffMs[this.reconnectAttempts] ?? 4000;
      this.reconnectAttempts++;
      console.error(
        `[engram-mcp] reconnecting in ${backoff}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`,
      );
      await sleep(backoff);
      try {
        await this.connect();
        return;
      } catch {
        // try next attempt
      }
    }
    throw new ConnectionLostError("exhausted reconnect attempts");
  }

  private handleData(chunk: Buffer): void {
    this.buffer += chunk.toString("utf8");
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const response: SocketResponse = JSON.parse(line);
        const entry = this.pending.get(response.id);
        if (!entry) continue;

        clearTimeout(entry.timer);
        this.pending.delete(response.id);

        if (response.ok) {
          entry.resolve(response.data);
        } else {
          entry.reject(
            new SocketError(
              response.error?.message ?? "unknown error",
              response.error?.code,
            ),
          );
        }
      } catch {
        console.error("[engram-mcp] failed to parse response:", line);
      }
    }
  }

  private rejectAllPending(reason: string): void {
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.reject(new SocketError(reason));
    }
    this.pending.clear();
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
