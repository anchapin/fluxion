import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ReconnectController,
  nextReconnectDelay,
} from "../src/livetwin/reconnect";

describe("nextReconnectDelay — capped exponential backoff", () => {
  it("waits the base delay before the first retry", () => {
    expect(nextReconnectDelay(1)).toBe(500);
  });

  it("grows exponentially with the attempt number", () => {
    expect(nextReconnectDelay(2)).toBe(1000);
    expect(nextReconnectDelay(3)).toBe(2000);
    expect(nextReconnectDelay(4)).toBe(4000);
  });

  it("never exceeds the cap", () => {
    expect(nextReconnectDelay(6)).toBe(15000);
    expect(nextReconnectDelay(20)).toBe(15000);
  });

  it("honours custom options", () => {
    expect(
      nextReconnectDelay(3, { baseDelayMs: 1000, factor: 3, maxDelayMs: 10_000 }),
    ).toBe(9000);
    expect(
      nextReconnectDelay(5, { baseDelayMs: 1000, factor: 3, maxDelayMs: 10_000 }),
    ).toBe(10_000);
  });

  it("treats non-positive attempt counts as the first retry", () => {
    expect(nextReconnectDelay(0)).toBe(500);
    expect(nextReconnectDelay(-7)).toBe(500);
  });
});

describe("ReconnectController", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("fires the retry after the scheduled backoff delay", () => {
    const ctl = new ReconnectController();
    const onReconnect = vi.fn();
    expect(ctl.schedule(onReconnect)).toBe(1);

    vi.advanceTimersByTime(499);
    expect(onReconnect).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(onReconnect).toHaveBeenCalledTimes(1);
    expect(onReconnect).toHaveBeenCalledWith(1, 500);
    expect(ctl.pending).toBe(false);
  });

  it("does not stack timers when a retry is already pending", () => {
    const ctl = new ReconnectController();
    const first = vi.fn();
    const second = vi.fn();
    expect(ctl.schedule(first)).toBe(1);
    expect(ctl.schedule(second)).toBeNull();

    vi.advanceTimersByTime(500);
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).not.toHaveBeenCalled();
  });

  it("backs off further on consecutive failures and resets after open", () => {
    const ctl = new ReconnectController();
    const cb = vi.fn();
    ctl.schedule(cb);
    vi.advanceTimersByTime(500); // attempt 1 fires

    expect(ctl.schedule(cb)).toBe(2);
    vi.advanceTimersByTime(999);
    expect(cb).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);
    expect(cb).toHaveBeenCalledWith(2, 1000);

    // Socket opens: next drop starts from the base delay again.
    ctl.recordOpen();
    expect(ctl.attempt).toBe(0);
    expect(ctl.schedule(cb)).toBe(1);
    vi.advanceTimersByTime(500);
    expect(cb).toHaveBeenCalledWith(1, 500);
  });

  it("cancel() drops a pending retry but keeps the attempt count", () => {
    const ctl = new ReconnectController();
    const cb = vi.fn();
    ctl.schedule(cb);
    ctl.schedule(cb); // ignored
    ctl.cancel();

    vi.advanceTimersByTime(60_000);
    expect(cb).not.toHaveBeenCalled();
    expect(ctl.pending).toBe(false);
    expect(ctl.attempt).toBe(1);

    // A later schedule continues the backoff sequence (attempt 2).
    expect(ctl.schedule(cb)).toBe(2);
  });

  it("reset() drops the retry and restarts the backoff", () => {
    const ctl = new ReconnectController();
    const cb = vi.fn();
    ctl.schedule(cb);
    vi.advanceTimersByTime(500);
    ctl.schedule(cb); // attempt 2
    ctl.reset();

    vi.advanceTimersByTime(60_000);
    expect(cb).toHaveBeenCalledTimes(1);
    expect(ctl.attempt).toBe(0);
    expect(ctl.schedule(cb)).toBe(1);
  });
});
