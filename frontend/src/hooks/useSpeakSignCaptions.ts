import { useEffect, useRef } from "react";
import type { Caption } from "./useRoom";

const STRIP_TRAILING_NUMBER = /[0-9]+$/;

/**
 * Speak each newly-arriving sign caption aloud via the Web Speech API so the
 * hearing user can hear what the signer is signing. Only captions added after
 * the hook mounts are spoken — historical captions in the buffer are skipped
 * to avoid a flood on first paint.
 *
 * Word-model labels from ASL Citizen carry an ASL-LEX disambiguation suffix
 * (e.g. ``fine1``, ``what1``). We strip the trailing digit before speaking so
 * the hearing user hears natural English.
 */
export function useSpeakSignCaptions(captions: Caption[], enabled: boolean = true) {
  const seenRef = useRef<Set<number>>(new Set());
  const firstRunRef = useRef(true);

  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    if (firstRunRef.current) {
      // Seed: mark current captions as "already seen" so we don't re-speak history.
      for (const c of captions) seenRef.current.add(c.id);
      firstRunRef.current = false;
      return;
    }
    if (!enabled) return;
    for (const c of captions) {
      if (seenRef.current.has(c.id)) continue;
      seenRef.current.add(c.id);
      if (c.source !== "sign") continue;
      const text = c.text.replace(STRIP_TRAILING_NUMBER, "").trim();
      if (!text) continue;
      try {
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.0;
        utter.pitch = 1.0;
        utter.lang = "en-US";
        window.speechSynthesis.speak(utter);
      } catch {
        // Browser may block speech synthesis without user gesture — fail silently.
      }
    }
  }, [captions, enabled]);
}
