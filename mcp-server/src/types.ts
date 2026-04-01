export interface SocketRequest {
  id: string;
  method: string;
  params: Record<string, unknown>;
}

export interface SocketResponse {
  id: string;
  ok: boolean;
  data?: unknown;
  error?: { code: number; message: string };
}

export enum LifecycleState {
  Disconnected = "disconnected",
  Spawning = "spawning",
  WaitingForSocket = "waiting_for_socket",
  Connected = "connected",
  Reconnecting = "reconnecting",
  Dead = "dead",
}

export interface SocketClientConfig {
  socketPath: string;
  connectTimeoutMs?: number;
  requestTimeoutMs?: number;
}

export interface LifecycleConfig {
  socketPath: string;
  engramBinary: string;
  spawnTimeoutMs?: number;
  healthIntervalMs?: number;
}

export class SocketError extends Error {
  constructor(message: string, public readonly code?: number) {
    super(message);
    this.name = "SocketError";
  }
}

export class ConnectionLostError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConnectionLostError";
  }
}
