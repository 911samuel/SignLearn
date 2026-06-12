"use client";

import { useEffect, useState } from "react";

const KEY = "signlearn.progress.v1";

export interface SignAttempt {
  sign: string;
  correct: boolean;
  confidence: number;
  ts: number;
}

export interface LessonProgress {
  lessonId: string;
  completed: boolean;
  bestScore: number;
  completedAt?: number;
}

export interface Progress {
  xp: number;
  streakDays: number;
  lastActiveDate?: string; // YYYY-MM-DD
  lessons: Record<string, LessonProgress>;
  attempts: SignAttempt[]; // last 200
}

const DEFAULTS: Progress = {
  xp: 0,
  streakDays: 0,
  lessons: {},
  attempts: [],
};

export function loadProgress(): Progress {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
  }
}

export function saveProgress(p: Progress) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(p));
  } catch {}
}

export function recordAttempt(sign: string, correct: boolean, confidence: number) {
  const p = loadProgress();
  const next: Progress = {
    ...p,
    xp: p.xp + (correct ? 10 : 2),
    attempts: [...p.attempts.slice(-199), { sign, correct, confidence, ts: Date.now() }],
  };
  // Streak: count distinct days touched.
  const today = new Date().toISOString().slice(0, 10);
  if (next.lastActiveDate !== today) {
    const yesterday = new Date(Date.now() - 86_400_000).toISOString().slice(0, 10);
    next.streakDays = next.lastActiveDate === yesterday ? p.streakDays + 1 : 1;
    next.lastActiveDate = today;
  }
  saveProgress(next);
  return next;
}

export function completeLesson(lessonId: string, score: number) {
  const p = loadProgress();
  const prev = p.lessons[lessonId];
  const next: Progress = {
    ...p,
    xp: p.xp + (prev?.completed ? 5 : 25),
    lessons: {
      ...p.lessons,
      [lessonId]: {
        lessonId,
        completed: true,
        bestScore: Math.max(prev?.bestScore ?? 0, score),
        completedAt: Date.now(),
      },
    },
  };
  saveProgress(next);
  return next;
}

export function clearProgress() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(KEY);
}

export function useProgress() {
  const [progress, setProgress] = useState<Progress>(DEFAULTS);

  useEffect(() => {
    setProgress(loadProgress());
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setProgress(loadProgress());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return progress;
}
