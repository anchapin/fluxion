/**
 * Reconnection policy for the LiveTwin WebSocket consumer (issue #3174).
 *
 * Pure, DOM-free logic so it is unit-testable under vitest's node
 * environment with fake timers — the hook (`useLiveTwin.ts`) only wires
 * this controller to WebSocket lifecycle events.
 */

export interface ReconnectOptions {
  /** Delay before the first retry. Default 500 ms. */
  baseDelayMs?: number;
  /** Upper bound for any single retry delay. Default 15 000 ms. */
  maxDelayMs?: number;
  /** Exponential growth factor between consecutive retries. Default 2. */
  factor?: number;
}

const DEFAULTS = {
  baseDelayMs: 500,
  maxDelayMs: 15_000,
  factor: 2,
} as const;

/**
 * Capped exponential backoff: attempt 1 waits `baseDelayMs`, attempt n
 * waits `baseDelayMs * factor^(n-1)`, never more than `maxDelayMs`.
 */
export function nextReconnectDelay(
  attempt: number,
  options: ReconnectOptions = {},
): number {
  const { baseDelayMs, maxDelayMs, factor } = { ...DEFAULTS, ...options };
  if (attempt <= 1) return Math.min(baseDelayMs, maxDelayMs);
  const raw = baseDelayMs * Math.pow(factor, attempt - 1);
  return Math.min(raw, maxDelayMs);
}

/**
 * Tracks consecutive failed connection attempts and owns the retry timer.
 * `useLiveTwin` drives it:
 * - `onclose` (not user-initiated)  -> `schedule(reconnect)`
 * - `onopen`                        -> `recordOpen()` (resets the backoff)
 * - manual `connect()`/`disconnect()` -> `reset()` / `cancel()`
 */
export class ReconnectController {
  private readonly options: Required<ReconnectOptions>;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private attemptCount = 0;

  constructor(options: ReconnectOptions = {}) {
    this.options = { ...DEFAULTS, ...options };
  }

  /** Number of consecutive failed attempts since the last open socket. */
  get attempt(): number {
    return this.attemptCount;
  }

  /** Whether a retry is currently pending. */
  get pending(): boolean {
    return this.timer !== null;
  }

  /**
   * Schedules one reconnect attempt. Returns the attempt number that will
   * be reported, or `null` when a retry is already pending (double
   * `onclose` events must not stack timers).
   */
  schedule(onReconnect: (attempt: number, delayMs: number) => void): number | null {
    if (this.timer !== null) return null;
    this.attemptCount += 1;
    const attempt = this.attemptCount;
    const delayMs = nextReconnectDelay(attempt, this.options);
    this.timer = setTimeout(() => {
      this.timer = null;
      onReconnect(attempt, delayMs);
    }, delayMs);
    return attempt;
  }

  /** Marks the connection healthy again — backoff restarts from the base. */
  recordOpen(): void {
    this.attemptCount = 0;
    this.clearTimer();
  }

  /** Drops a pending retry without touching the attempt counter. */
  cancel(): void {
    this.clearTimer();
  }

  /** Drops a pending retry and restarts the backoff sequence. */
  reset(): void {
    this.attemptCount = 0;
    this.clearTimer();
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
