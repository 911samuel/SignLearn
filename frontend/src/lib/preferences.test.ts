import { describe, it, expect, beforeEach } from "vitest";
import { loadPreferences, savePreferences, applyPreferences } from "./preferences";

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("data-text-size");
  document.documentElement.removeAttribute("data-reduce-motion");
});

describe("loadPreferences", () => {
  it("returns defaults when nothing is stored", () => {
    expect(loadPreferences()).toMatchObject({
      theme: "system",
      textSize: "normal",
      reduceMotion: false,
      captionSize: "large",
      pushToTalkMode: "hold",
      name: "",
    });
  });

  it("merges stored partial over defaults", () => {
    window.localStorage.setItem(
      "signlearn.prefs.v1",
      JSON.stringify({ theme: "dark", name: "Sam" }),
    );
    const p = loadPreferences();
    expect(p.theme).toBe("dark");
    expect(p.name).toBe("Sam");
    expect(p.textSize).toBe("normal");
  });

  it("falls back to defaults on malformed JSON", () => {
    window.localStorage.setItem("signlearn.prefs.v1", "{not json");
    expect(loadPreferences().theme).toBe("system");
  });
});

describe("savePreferences", () => {
  it("round-trips through localStorage", () => {
    const prefs = {
      theme: "dark" as const,
      textSize: "large" as const,
      reduceMotion: true,
      captionSize: "xlarge" as const,
      pushToTalkMode: "toggle" as const,
      name: "Ada",
    };
    savePreferences(prefs);
    expect(loadPreferences()).toEqual(prefs);
  });
});

describe("applyPreferences", () => {
  it("sets data-theme except for 'system'", () => {
    applyPreferences({
      theme: "dark",
      textSize: "normal",
      reduceMotion: false,
      captionSize: "normal",
      pushToTalkMode: "hold",
      name: "",
    });
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");

    applyPreferences({
      theme: "system",
      textSize: "normal",
      reduceMotion: false,
      captionSize: "normal",
      pushToTalkMode: "hold",
      name: "",
    });
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });

  it("toggles data-reduce-motion", () => {
    applyPreferences({
      theme: "system",
      textSize: "normal",
      reduceMotion: true,
      captionSize: "normal",
      pushToTalkMode: "hold",
      name: "",
    });
    expect(document.documentElement.getAttribute("data-reduce-motion")).toBe("true");
  });

  it("sets data-text-size only when not 'normal'", () => {
    applyPreferences({
      theme: "system",
      textSize: "xlarge",
      reduceMotion: false,
      captionSize: "normal",
      pushToTalkMode: "hold",
      name: "",
    });
    expect(document.documentElement.getAttribute("data-text-size")).toBe("xlarge");
  });
});
