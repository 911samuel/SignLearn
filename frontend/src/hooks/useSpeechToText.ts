import { useState, useRef, useCallback, useEffect } from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:5001";

export interface SpeechEntry {
  id: number;
  text: string;
  ts: number;
}

export function useSpeechToText(onResult?: (text: string, ts: number) => void) {
  const [transcript, setTranscript] = useState<SpeechEntry[]>([]);
  const [listening, setListening] = useState(false);
  const [supported] = useState(
    () =>
      typeof window !== "undefined" &&
      ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recogRef = useRef<any>(null);
  const idRef = useRef(0);
  const onResultRef = useRef(onResult);
  useEffect(() => { onResultRef.current = onResult; }, [onResult]);

  const start = useCallback(() => {
    if (!supported || listening) return;

    // Vendor-prefixed fallback for Safari/Chrome
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SR: new () => any =
      (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;

    const recog = new SR();
    recog.continuous = true;
    recog.interimResults = false;
    recog.lang = "en-US";

    recog.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) {
          const text = e.results[i][0].transcript.trim();
          if (text) {
            const ts = Date.now();
            setTranscript((prev) => [
              ...prev,
              { id: ++idRef.current, text, ts },
            ]);
            onResultRef.current?.(text, ts);
            // Persist to backend (fire-and-forget)
            fetch(`${BACKEND_URL}/speech-to-text`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text }),
            }).catch((err) => {
              console.warn("[useSpeechToText] Failed to persist speech entry:", err);
            });
          }
        }
      }
    };

    recog.onend = () => setListening(false);

    recogRef.current = recog;
    recog.start();
    setListening(true);
  }, [supported, listening]);

  const stop = useCallback(() => {
    recogRef.current?.stop();
    setListening(false);
  }, []);

  const clear = useCallback(() => setTranscript([]), []);

  return { transcript, listening, supported, start, stop, clear };
}
