import { useState, useRef, useCallback, useEffect } from "react";

export interface SpeechEntry {
  id: number;
  text: string;
  ts: number;
}

/**
 * Wraps the browser's Web Speech API. The `onResult` callback receives every
 * finalized utterance; callers typically forward it to `useRoom.emitSpeech`
 * so the caption is broadcast to the peer.
 */
export function useSpeechToText(onResult?: (text: string, ts: number) => void) {
  const [transcript, setTranscript] = useState<SpeechEntry[]>([]);
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    setSupported(
      typeof window !== "undefined" &&
        ("SpeechRecognition" in window || "webkitSpeechRecognition" in window),
    );
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recogRef = useRef<any>(null);
  const idRef = useRef(0);
  const onResultRef = useRef(onResult);
  useEffect(() => {
    onResultRef.current = onResult;
  }, [onResult]);

  const start = useCallback(() => {
    if (!supported || listening) return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SR: new () => any =
      (window as any).SpeechRecognition ??
      (window as any).webkitSpeechRecognition;

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
