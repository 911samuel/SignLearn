import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  loadProgress,
  saveProgress,
  recordAttempt,
  completeLesson,
  clearProgress,
} from "./progress";

beforeEach(() => {
  window.localStorage.clear();
  vi.useRealTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("loadProgress", () => {
  it("returns defaults when empty", () => {
    const p = loadProgress();
    expect(p).toEqual({ xp: 0, streakDays: 0, lessons: {}, attempts: [] });
  });

  it("falls back to defaults on malformed JSON", () => {
    window.localStorage.setItem("signlearn.progress.v1", "garbage");
    expect(loadProgress().xp).toBe(0);
  });
});

describe("recordAttempt", () => {
  it("awards 10 xp on correct, 2 on wrong", () => {
    const a = recordAttempt("a", true, 0.9);
    expect(a.xp).toBe(10);
    const b = recordAttempt("b", false, 0.4);
    expect(b.xp).toBe(12);
  });

  it("appends to attempts and caps at 200", () => {
    const seed = { xp: 0, streakDays: 0, lessons: {}, attempts: [] as any[] };
    for (let i = 0; i < 205; i++) {
      seed.attempts.push({ sign: "x", correct: true, confidence: 1, ts: i });
    }
    saveProgress(seed as any);
    const next = recordAttempt("new", true, 0.5);
    expect(next.attempts.length).toBe(200);
    expect(next.attempts[next.attempts.length - 1].sign).toBe("new");
  });

  it("starts streak at 1 on first activity", () => {
    const p = recordAttempt("a", true, 1);
    expect(p.streakDays).toBe(1);
  });

  it("increments streak when previous activity was yesterday", () => {
    const yesterday = new Date(Date.now() - 86_400_000).toISOString().slice(0, 10);
    saveProgress({
      xp: 0,
      streakDays: 3,
      lastActiveDate: yesterday,
      lessons: {},
      attempts: [],
    });
    const p = recordAttempt("a", true, 1);
    expect(p.streakDays).toBe(4);
  });

  it("resets streak to 1 when there is a gap", () => {
    saveProgress({
      xp: 0,
      streakDays: 7,
      lastActiveDate: "2020-01-01",
      lessons: {},
      attempts: [],
    });
    const p = recordAttempt("a", true, 1);
    expect(p.streakDays).toBe(1);
  });

  it("does not bump streak twice on the same day", () => {
    const first = recordAttempt("a", true, 1);
    const second = recordAttempt("b", true, 1);
    expect(second.streakDays).toBe(first.streakDays);
  });
});

describe("completeLesson", () => {
  it("awards 25 xp the first time and 5 xp on re-completion", () => {
    const a = completeLesson("alphabet-1", 80);
    expect(a.xp).toBe(25);
    expect(a.lessons["alphabet-1"].bestScore).toBe(80);

    const b = completeLesson("alphabet-1", 60);
    expect(b.xp).toBe(30);
    expect(b.lessons["alphabet-1"].bestScore).toBe(80);
  });

  it("keeps the higher best score", () => {
    completeLesson("l", 50);
    const p = completeLesson("l", 95);
    expect(p.lessons["l"].bestScore).toBe(95);
  });
});

describe("clearProgress", () => {
  it("removes the stored key", () => {
    recordAttempt("a", true, 1);
    clearProgress();
    expect(window.localStorage.getItem("signlearn.progress.v1")).toBeNull();
    expect(loadProgress().xp).toBe(0);
  });
});
