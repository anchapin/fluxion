import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_LIVE_TWIN_URL,
  decodeLiveTwinFrame,
  zoneNumber,
  type LiveTwinPayload,
} from "./protocol";
import { ReconnectController } from "./reconnect";

export type LiveTwinStatus =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "error";

export interface LiveTwinState {
  status: LiveTwinStatus;
  url: string;
  /** Latest zone state keyed by numeric zone id (LiveTwin `zone_id`). */
  zones: Map<number, LiveTwinPayload["zone_states"][0]>;
  payloadCount: number;
  lastError: string | null;
  /** Consecutive reconnect attempt number (0 = none / connected). */
  reconnectAttempt: number;
  connect: (url?: string) => void;
  disconnect: () => void;
}

/**
 * LiveTwin WebSocket hook. Connects to the MessagePack stream exposed by
 * `src/twin/live_twin_broadcaster.rs` (default ws://localhost:8080/live-twin)
 * and keeps the latest per-zone state. Works identically in Tauri and web
 * mode — any browser context can reach a running simulation backend.
 *
 * Issue #3174: dropped sockets are retried with capped exponential backoff
 * (500 ms → 15 s) until `disconnect()` is called or the component unmounts;
 * a successful `onopen` resets the backoff sequence.
 */
export function useLiveTwin(): LiveTwinState {
  const [status, setStatus] = useState<LiveTwinStatus>("idle");
  const [url, setUrl] = useState(DEFAULT_LIVE_TWIN_URL);
  const [zones, setZones] = useState(
    () => new Map<number, LiveTwinPayload["zone_states"][0]>(),
  );
  const [payloadCount, setPayloadCount] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const urlRef = useRef(DEFAULT_LIVE_TWIN_URL);
  const manualCloseRef = useRef(false);
  const reconnectRef = useRef(new ReconnectController());

  const openSocket = useCallback((target: string) => {
    socketRef.current?.close();
    setStatus("connecting");

    let ws: WebSocket;
    try {
      ws = new WebSocket(target);
    } catch (err) {
      setStatus("error");
      setLastError(String(err));
      return;
    }
    // rmp-serde sends binary frames.
    ws.binaryType = "arraybuffer";
    socketRef.current = ws;

    ws.onopen = () => {
      reconnectRef.current.recordOpen();
      setReconnectAttempt(0);
      setStatus("open");
    };
    ws.onclose = () => {
      if (socketRef.current !== ws) return;
      socketRef.current = null;
      if (manualCloseRef.current) {
        setStatus("closed");
        return;
      }
      // Unexpected drop (server restart, network loss, refused initial
      // connect): retry with exponential backoff until it comes back.
      setStatus("reconnecting");
      reconnectRef.current.schedule((attempt) => {
        setReconnectAttempt(attempt);
        if (!manualCloseRef.current) openSocket(urlRef.current);
      });
    };
    ws.onerror = () => {
      setStatus("error");
      setLastError(`WebSocket error for ${target}`);
    };
    ws.onmessage = (event: MessageEvent) => {
      const payload = decodeLiveTwinFrame(event.data);
      if (!payload) return;
      setZones((prev) => {
        const next = new Map(prev);
        for (const z of payload.zone_states) {
          next.set(zoneNumber(z.zone_id), z);
        }
        return next;
      });
      setPayloadCount((n) => n + 1);
    };
  }, []);

  const disconnect = useCallback(() => {
    manualCloseRef.current = true;
    reconnectRef.current.cancel();
    socketRef.current?.close();
    socketRef.current = null;
    setStatus("closed");
  }, []);

  const connect = useCallback((nextUrl?: string) => {
    const target = nextUrl ?? urlRef.current;
    manualCloseRef.current = false;
    reconnectRef.current.reset();
    setReconnectAttempt(0);
    urlRef.current = target;
    setUrl(target);
    setLastError(null);
    openSocket(target);
  }, [openSocket]);

  useEffect(() => disconnect, [disconnect]);

  return {
    status,
    url,
    zones,
    payloadCount,
    lastError,
    reconnectAttempt,
    connect,
    disconnect,
  };
}
