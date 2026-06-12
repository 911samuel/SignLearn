"use client";

import { useEffect, useState } from "react";

export type Theme = "light" | "dark" | "high-contrast" | "system";
export type TextSize = "normal" | "large" | "xlarge";

export interface Preferences {
  theme: Theme;
  textSize: TextSize;
  reduceMotion: boolean;
  captionSize: "normal" | "large" | "xlarge";
  pushToTalkMode: "hold" | "toggle";
  /** Display name used in rooms. Empty until the user enters it. */
  name: string;
}

const KEY = "signlearn.prefs.v1";

const DEFAULTS: Preferences = {
  theme: "system",
  textSize: "normal",
  reduceMotion: false,
  captionSize: "large",
  pushToTalkMode: "hold",
  name: "",
};

export function loadPreferences(): Preferences {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
  }
}

export function savePreferences(prefs: Preferences) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    /* quota or disabled */
  }
}

export function applyPreferences(prefs: Preferences) {
  if (typeof document === "undefined") return;
  const html = document.documentElement;

  if (prefs.theme === "system") {
    html.removeAttribute("data-theme");
  } else {
    html.setAttribute("data-theme", prefs.theme);
  }

  if (prefs.textSize === "normal") {
    html.removeAttribute("data-text-size");
  } else {
    html.setAttribute("data-text-size", prefs.textSize);
  }

  if (prefs.reduceMotion) {
    html.setAttribute("data-reduce-motion", "true");
  } else {
    html.removeAttribute("data-reduce-motion");
  }
}

export function usePreferences() {
  const [prefs, setPrefs] = useState<Preferences>(DEFAULTS);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const loaded = loadPreferences();
    setPrefs(loaded);
    applyPreferences(loaded);
    setHydrated(true);
  }, []);

  const update = (next: Partial<Preferences>) => {
    setPrefs((cur) => {
      const merged = { ...cur, ...next };
      savePreferences(merged);
      applyPreferences(merged);
      return merged;
    });
  };

  return { prefs, update, hydrated };
}
