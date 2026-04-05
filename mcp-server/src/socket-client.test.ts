import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { EventEmitter } from "node:events";
import { SocketError } from "./types.js";

class MockSocket extends EventEmitter {
  writable = true;
  destroyed = false;

  write(data: string, _encoding: string, callback?: (error?: Error) => void): boolean {
    if (callback) callback();
    return true;
  }

  end(callback?: () => void): this {
    this.writable = false;
    if (callback) callback();
    return this;
  }

  destroy(): this {
    this.destroyed = true;
    this.writable = false;
    return this;
  }
}

let mockSocket: MockSocket;

vi.mock("node:net", () => ({
  createConnection: () => mockSocket,
}));

const { SocketClient } = await import("./socket-client.js");

describe("SocketClient", () => {
  beforeEach(() => {
    mockSocket = new MockSocket();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("connect() resolves on socket connect event", async () => {
    const client = new SocketClient({ socketPath: "/tmp/test.sock" });

    const connectPromise = client.connect();
    mockSocket.emit("connect");

    await expect(connectPromise).resolves.toBeUndefined();
  });

  it("connect() rejects with SocketError on timeout", async () => {
    const client = new SocketClient({
      socketPath: "/tmp/test.sock",
      connectTimeoutMs: 50,
    });

    const connectPromise = client.connect();
    vi.advanceTimersByTime(50);

    await expect(connectPromise).rejects.toThrow(SocketError);
    await expect(connectPromise).rejects.toThrow(/timeout/i);
  });

  it("call() resolves with data on successful response", async () => {
    const client = new SocketClient({ socketPath: "/tmp/test.sock" });

    const connectPromise = client.connect();
    mockSocket.emit("connect");
    await connectPromise;

    const writeOriginal = mockSocket.write.bind(mockSocket);
    mockSocket.write = (data: string, encoding: string, callback?: (error?: Error) => void): boolean => {
      writeOriginal(data, encoding, callback);
      const request = JSON.parse(data.trim());
      const response = JSON.stringify({ id: request.id, ok: true, data: { status: "healthy" } });
      process.nextTick(() => mockSocket.emit("data", Buffer.from(response + "\n")));
      return true;
    };

    const result = await client.call("memory_status", {});
    expect(result).toEqual({ status: "healthy" });
  });

  it("call() rejects with SocketError on error response", async () => {
    const client = new SocketClient({ socketPath: "/tmp/test.sock" });

    const connectPromise = client.connect();
    mockSocket.emit("connect");
    await connectPromise;

    const writeOriginal = mockSocket.write.bind(mockSocket);
    mockSocket.write = (data: string, encoding: string, callback?: (error?: Error) => void): boolean => {
      writeOriginal(data, encoding, callback);
      const request = JSON.parse(data.trim());
      const response = JSON.stringify({
        id: request.id,
        ok: false,
        error: { code: 404, message: "method not found" },
      });
      process.nextTick(() => mockSocket.emit("data", Buffer.from(response + "\n")));
      return true;
    };

    await expect(client.call("unknown_method", {})).rejects.toThrow(SocketError);
    await expect(client.call("unknown_method", {})).rejects.toThrow("method not found");
  });

  it("call() rejects with SocketError on request timeout", async () => {
    const client = new SocketClient({
      socketPath: "/tmp/test.sock",
      requestTimeoutMs: 50,
    });

    const connectPromise = client.connect();
    mockSocket.emit("connect");
    await connectPromise;

    const callPromise = client.call("slow_method", {});
    vi.advanceTimersByTime(50);

    await expect(callPromise).rejects.toThrow(SocketError);
    await expect(callPromise).rejects.toThrow(/timeout/i);
  });

  it("close() rejects all pending requests", async () => {
    const client = new SocketClient({ socketPath: "/tmp/test.sock" });

    const connectPromise = client.connect();
    mockSocket.emit("connect");
    await connectPromise;

    const callPromise = client.call("pending_method", {});
    await client.close();

    await expect(callPromise).rejects.toThrow(SocketError);
    await expect(callPromise).rejects.toThrow("client closing");
  });

  it("close() is safe to call multiple times", async () => {
    const client = new SocketClient({ socketPath: "/tmp/test.sock" });

    const connectPromise = client.connect();
    mockSocket.emit("connect");
    await connectPromise;

    await client.close();
    await expect(client.close()).resolves.toBeUndefined();
  });
});
