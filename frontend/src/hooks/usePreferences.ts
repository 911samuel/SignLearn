import { useCallback, useEffect, useState } from "react";

export type TextSize = "normal" | "large";

export interface Preferences {
  name: string;
  textSize: TextSize;
  confidenceThreshold: number; // 0..1
  soundEnabled: boolean;
}

const DEFAULTS: Preferences = {
  name: "",
  textSize: "normal",
  confidenceThreshold: 0.7,
  soundEnabled: false,
};

const KEY = "sl-prefs";

function load(): Preferences {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
  }
}

function save(p: Preferences) {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {}
}

export function usePreferences(): [
  Preferences,
  (patch: Partial<Preferences>) => void,
] {
  const [prefs, setPrefs] = useState<Preferences>(DEFAULTS);

  useEffect(() => {
    setPrefs(load());
  }, []);

  // Apply text-size to :root whenever it changes.
  useEffect(() => {
    document.documentElement.setAttribute("data-text-size", prefs.textSize);
  }, [prefs.textSize]);

  const update = useCallback((patch: Partial<Preferences>) => {
    setPrefs((prev) => {
      const next = { ...prev, ...patch };
      save(next);
      return next;
    });
  }, []);

  return [prefs, update];
}
