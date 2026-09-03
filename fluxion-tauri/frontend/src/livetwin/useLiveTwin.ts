import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_LIVE_TWIN_URL,
  decodeLiveTwinFrame,
  zoneNumber,
  type LiveTwinPayload,
} from "./protocol";

export type LiveTwinStatus = "idle" | "connecting" | "open" | "closed" | "error";

export interface LiveTwinState {
  status: LiveTwinStatus;
  url: string;
  /** Latest zone state keyed by numeric zone id (LiveTwin `zone_id`). */
  zones: Map<number, LiveTwinPayload["zone_states"][0]>;
  payloadCount: number;
  lastError: string | null;
  connect: (url?: string) => void;
  disconnect: () => void;
}

/**
 * LiveTwin WebSocket hook. Connects to the MessagePack stream exposed by
 * `src/twin/live_twin_broadcaster.rs` (default ws://localhost:8080/live-twin)
 * and keeps the latest per-zone state. Works identically in Tauri and web
 * mode — any browser context can reach a running simulation backend.
 */
export function useLiveTwin(): LiveTwinState {
  const [status, setStatus] = useState<LiveTwinStatus>("idle");
  const [url, setUrl] = useState(DEFAULT_LIVE_TWIN_URL);
  const [zones, setZones] = useState(
    () => new Map<number, LiveTwinPayload["zone_states"][0]>(),
  );
  const [payloadCount, setPayloadCount] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setStatus("closed");
  }, []);

  const connect = useCallback(
    (nextUrl?: string) => {
      const target = nextUrl ?? url;
      socketRef.current?.close();
      setUrl(target);
      setStatus("connecting");
      setLastError(null);

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

      ws.onopen = () => setStatus("open");
      ws.onclose = () => {
        if (socketRef.current === ws) {
          socketRef.current = null;
          setStatus("closed");
        }
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
    },
    [url],
  );

  useEffect(() => disconnect, [disconnect]);

  return { status, url, zones, payloadCount, lastError, connect, disconnect };
}
