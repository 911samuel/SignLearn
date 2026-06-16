import { useEffect, useRef, useCallback, useState } from "react";
import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type { Socket } from "socket.io-client";

const FEATURE_DIM = 126; // 2 hands × 21 landmarks × 3 coords

export interface Prediction {
  label: string | null;
  confidence: number | null;
  ready: boolean;
}

export interface WordCandidate {
  label: string;
  confidence: number;
}

export interface WordPrediction {
  top3: WordCandidate[];
  error?: string;
}

export type { ConnectionStatus } from "./useRoom";

/**
 * Runs MediaPipe hand-landmark extraction in-browser and streams the
 * resulting 126-float vectors over an externally-supplied Socket.IO
 * connection. Prediction events come back through the same socket.
 *
 * `socket` may be null on first render (before the room hook connects);
 * the effect simply waits until it's available.
 */
export function useSignRecognition(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  socket: Socket | null,
) {
  const [prediction, setPrediction] = useState<Prediction>({
    label: null,
    confidence: null,
    ready: false,
  });
  // Unified utterance — accumulates BOTH word predictions and fingerspelled
  // letters in the order they arrive.  Letters within the same finger-spell
  // run concatenate (S, A, M → "SAM"); words and letter-runs are separated
  // by a single space ("hello SAM coffee").  Closes when the signer drops
  // their hands out of frame (end-of-turn signal in ASL), on manual reset,
  // or after IDLE_CLEAR_MS of nothing arriving.
  const [utterance, setUtterance] = useState<string>("");
  // When an utterance closes, it briefly appears as `sentUtterance` so the
  // signer sees a "Sent" confirmation before the row clears.
  const [sentUtterance, setSentUtterance] = useState<string | null>(null);
  const lastLetterRef = useRef<string | null>(null);
  const lastWasLetterRef = useRef<boolean>(false);
  const utteranceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sentFlashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const framesWithoutHandRef = useRef<number>(0);
  // ~2 seconds of hand-out-of-frame at 30fps = end of utterance.
  const HAND_DOWN_FRAMES_TO_CLOSE = 60;
  const SENT_FLASH_MS = 1500;
  const IDLE_CLEAR_MS = 10_000;

  const scheduleUtteranceClear = useCallback(() => {
    if (utteranceTimerRef.current) clearTimeout(utteranceTimerRef.current);
    utteranceTimerRef.current = setTimeout(() => {
      setUtterance("");
      lastLetterRef.current = null;
      lastWasLetterRef.current = false;
    }, IDLE_CLEAR_MS);
  }, []);

  // Commit the current utterance as "done" — drops the hands signal in ASL.
  // Flashes the finalized text as `sentUtterance` then clears the row so
  // the next utterance starts fresh.  Per-word captions have already been
  // streamed to the room via the backend's caption emit; this gives the
  // signer (and the conversation log) a visible end-of-sentence boundary.
  const commitUtterance = useCallback(() => {
    setUtterance((prev) => {
      if (!prev) return prev;
      // Snapshot the text into the flash slot and tell the room.
      setSentUtterance(prev);
      const sock = socketRef.current;
      if (sock) sock.emit("utterance_complete", { text: prev, t: Date.now() });
      if (sentFlashTimerRef.current) clearTimeout(sentFlashTimerRef.current);
      sentFlashTimerRef.current = setTimeout(
        () => setSentUtterance(null),
        SENT_FLASH_MS,
      );
      lastLetterRef.current = null;
      lastWasLetterRef.current = false;
      if (utteranceTimerRef.current) {
        clearTimeout(utteranceTimerRef.current);
        utteranceTimerRef.current = null;
      }
      return "";
    });
  }, []);
  const [landmarkerResult, setLandmarkerResult] = useState<HandLandmarkerResult | null>(null);
  const [paused, setPaused] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [landmarkerError, setLandmarkerError] = useState<string | null>(null);

  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const pausedRef = useRef(false);
  const lastFrameTsRef = useRef<number>(0);
  // Keep a ref to socket so the rAF loop always uses the current value.
  const socketRef = useRef<Socket | null>(socket);
  useEffect(() => { socketRef.current = socket; }, [socket]);

  // Word-capture state — automatically segments signs via motion energy.
  // State machine: idle → signing → complete → idle.
  // No user button: when hand motion crosses HIGH threshold the buffer fills,
  // when motion stays below LOW threshold for IDLE_FRAMES_TO_COMPLETE the
  // buffered sequence is sent to the word model.
  const [wordPrediction, setWordPrediction] = useState<WordPrediction | null>(null);
  const [captureStatus, setCaptureStatus] = useState<"idle" | "signing" | "processing">("idle");
  const [captureProgress, setCaptureProgress] = useState(0); // 0..1 of WORD_MAX_FRAMES
  const wordBufferRef = useRef<number[][]>([]);
  const prevLandmarksRef = useRef<number[] | null>(null);
  const idleFramesRef = useRef(0);
  const stateRef = useRef<"idle" | "signing" | "processing">("idle");
  const WORD_MAX_FRAMES = 80;
  const MIN_FRAMES_BEFORE_PREDICT = 15;   // ignore tiny twitches
  const MOTION_HIGH = 0.015;              // displacement to start signing
  const MOTION_LOW = 0.005;               // displacement considered "still"
  const IDLE_FRAMES_TO_COMPLETE = 8;      // ~270ms at 30fps before predicting

  // Initialise MediaPipe HandLandmarker
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        if (!cancelled) landmarkerRef.current = landmarker;
      } catch (err) {
        if (!cancelled) {
          setLandmarkerError(
            err instanceof Error ? err.message : "Failed to load MediaPipe — check your network connection."
          );
        }
      }
    }

    init();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  // Listen for word_prediction events; flip the capture state machine back to
  // idle and append the top word to the unified utterance.
  useEffect(() => {
    if (!socket) return;
    const wordHandler = (data: WordPrediction) => {
      setWordPrediction(data);
      if (lastFrameTsRef.current) {
        setLatencyMs(Date.now() - lastFrameTsRef.current);
      }
      stateRef.current = "idle";
      setCaptureStatus("idle");
      setCaptureProgress(0);
      if (!data.error) {
        const top = data.top3?.[0]?.label;
        if (top) {
          setUtterance((prev) => (prev ? `${prev} ${top}` : top));
          lastWasLetterRef.current = false;
          scheduleUtteranceClear();
        }
      }
    };
    socket.on("word_prediction", wordHandler);
    return () => {
      socket.off("word_prediction", wordHandler);
    };
  }, [socket, scheduleUtteranceClear]);

  // Listen for letter/digit prediction events.  The backend's letter model
  // accumulates the last 30 frames server-side and emits a smoothed result;
  // we subscribe and append distinct labels onto the unified utterance.
  // Consecutive letters in the same fingerspelling run concatenate (S, A, M
  // → "SAM"); a letter following a word gets a leading space.
  useEffect(() => {
    if (!socket) return;
    const letterHandler = (data: Prediction) => {
      setPrediction(data);
      if (data.ready && data.label && data.label !== lastLetterRef.current) {
        lastLetterRef.current = data.label;
        // Letters and digits both come through here; uppercase letters for
        // visual rhythm ("SAM" reads better than "sam").
        const ch = /^[a-z]$/.test(data.label)
          ? data.label.toUpperCase()
          : data.label;
        setUtterance((prev) => {
          if (!prev) return ch;
          // Continuing a fingerspell run: glue letters together.
          // Starting a new letter run after a word: add a separator space.
          return lastWasLetterRef.current ? prev + ch : `${prev} ${ch}`;
        });
        lastWasLetterRef.current = true;
        scheduleUtteranceClear();
      }
    };
    socket.on("prediction", letterHandler);
    return () => {
      socket.off("prediction", letterHandler);
    };
  }, [socket, scheduleUtteranceClear]);

  // rAF loop: extract landmarks → motion-gated auto-segment → emit word_predict
  useEffect(() => {
    function setStatus(s: "idle" | "signing" | "processing") {
      stateRef.current = s;
      setCaptureStatus(s);
    }

    function tick() {
      rafRef.current = requestAnimationFrame(tick);
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      const sock = socketRef.current;

      if (!video || !landmarker || video.readyState < 2) return;
      if (video.currentTime === lastVideoTimeRef.current) return;
      lastVideoTimeRef.current = video.currentTime;

      const result: HandLandmarkerResult = landmarker.detectForVideo(
        video,
        performance.now()
      );
      setLandmarkerResult(result);
      if (pausedRef.current) return;

      const landmarks = flattenLandmarks(result);
      const hasHand = landmarks.some((v) => v !== 0);

      // End-of-utterance detection — when the signer drops their hands out
      // of frame for HAND_DOWN_FRAMES_TO_CLOSE consecutive frames (~2s),
      // finalize whatever's accumulated so far.  Only triggers while the
      // word capture state machine is idle so we don't kill an in-flight
      // word capture from the user re-positioning their hand briefly.
      if (!hasHand) {
        framesWithoutHandRef.current++;
        if (
          framesWithoutHandRef.current === HAND_DOWN_FRAMES_TO_CLOSE &&
          stateRef.current === "idle"
        ) {
          commitUtterance();
        }
      } else {
        framesWithoutHandRef.current = 0;
      }

      // Letter/digit pipeline — stream every frame to the backend, but
      // ONLY while the word capture state machine is idle.  During a word
      // capture, the same moving hand would otherwise feed the letter
      // sliding window and produce nonsense letter predictions that
      // collide with the word prediction.  Suppressing here keeps the
      // letter overlay quiet during words and active during stillness
      // (fingerspelling) — the natural ASL boundary.
      if (sock && hasHand && stateRef.current === "idle") {
        sock.emit("frame", { landmarks, t: Date.now() });
      }

      // Compute frame-to-frame mean landmark displacement (motion energy).
      let motion = 0;
      if (prevLandmarksRef.current && hasHand) {
        let sum = 0;
        let n = 0;
        for (let i = 0; i < landmarks.length; i++) {
          if (landmarks[i] !== 0 && prevLandmarksRef.current[i] !== 0) {
            sum += Math.abs(landmarks[i] - prevLandmarksRef.current[i]);
            n++;
          }
        }
        motion = n > 0 ? sum / n : 0;
      }
      prevLandmarksRef.current = landmarks;

      const state = stateRef.current;

      if (state === "idle") {
        if (hasHand && motion >= MOTION_HIGH) {
          wordBufferRef.current = [landmarks];
          idleFramesRef.current = 0;
          setStatus("signing");
          setCaptureProgress(1 / WORD_MAX_FRAMES);
          // Hand the channel over to the word pipeline: flush the server
          // letter sliding window and the in-flight letter pill so the
          // pending letter doesn't get appended on top of the incoming
          // word.  The unified utterance is preserved — letters already
          // spelled stay in the row, the new word will append to them.
          if (sock) sock.emit("reset");
          setPrediction({ label: null, confidence: null, ready: false });
          lastLetterRef.current = null;
        }
        return;
      }

      if (state === "signing") {
        // Keep buffering
        if (wordBufferRef.current.length < WORD_MAX_FRAMES) {
          wordBufferRef.current.push(landmarks);
          setCaptureProgress(wordBufferRef.current.length / WORD_MAX_FRAMES);
        }
        // Idle detection
        if (!hasHand || motion < MOTION_LOW) {
          idleFramesRef.current++;
        } else {
          idleFramesRef.current = 0;
        }
        // Commit if we hit max frames or sustained stillness
        const enoughIdle = idleFramesRef.current >= IDLE_FRAMES_TO_COMPLETE;
        const atCapacity = wordBufferRef.current.length >= WORD_MAX_FRAMES;
        if (enoughIdle || atCapacity) {
          const frames = wordBufferRef.current;
          wordBufferRef.current = [];
          idleFramesRef.current = 0;
          if (frames.length < MIN_FRAMES_BEFORE_PREDICT) {
            // Too-short twitch; drop it silently.
            setStatus("idle");
            setCaptureProgress(0);
            return;
          }
          if (sock) {
            lastFrameTsRef.current = Date.now();
            sock.emit("word_predict", { frames });
            setStatus("processing");
            // Auto-return to idle after a short window if no response arrives
            setTimeout(() => {
              if (stateRef.current === "processing") {
                setStatus("idle");
                setCaptureProgress(0);
              }
            }, 2500);
          } else {
            setStatus("idle");
            setCaptureProgress(0);
          }
        }
        return;
      }

      // state === "processing" — wait for backend response; ignored here.
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [videoRef, commitUtterance]);

  const reset = useCallback(() => {
    wordBufferRef.current = [];
    idleFramesRef.current = 0;
    prevLandmarksRef.current = null;
    stateRef.current = "idle";
    setCaptureStatus("idle");
    setCaptureProgress(0);
    setUtterance("");
    setSentUtterance(null);
    lastLetterRef.current = null;
    lastWasLetterRef.current = false;
    framesWithoutHandRef.current = 0;
    if (utteranceTimerRef.current) {
      clearTimeout(utteranceTimerRef.current);
      utteranceTimerRef.current = null;
    }
    if (sentFlashTimerRef.current) {
      clearTimeout(sentFlashTimerRef.current);
      sentFlashTimerRef.current = null;
    }
    setPrediction({ label: null, confidence: null, ready: false });
    setWordPrediction(null);
  }, []);

  const togglePaused = useCallback(() => {
    setPaused((p) => {
      const next = !p;
      pausedRef.current = next;
      // If pausing mid-capture, drop the buffer to avoid emitting a garbled
      // prediction when the user resumes.
      if (next) {
        wordBufferRef.current = [];
        idleFramesRef.current = 0;
        stateRef.current = "idle";
        setCaptureStatus("idle");
        setCaptureProgress(0);
      }
      return next;
    });
  }, []);

  return {
    prediction,
    utterance,
    sentUtterance,
    commitUtterance,
    wordPrediction,
    captureStatus,
    captureProgress,
    landmarkerResult,
    reset,
    paused,
    togglePaused,
    latencyMs,
    landmarkerError,
  };
}

// Flatten up to 2 hands into a fixed-length 126-float array.
function flattenLandmarks(result: HandLandmarkerResult): number[] {
  const out = new Array<number>(FEATURE_DIM).fill(0);
  for (let h = 0; h < Math.min(result.landmarks.length, 2); h++) {
    const base = h * 63;
    for (let l = 0; l < 21; l++) {
      const lm = result.landmarks[h][l];
      out[base + l * 3] = lm.x;
      out[base + l * 3 + 1] = lm.y;
      out[base + l * 3 + 2] = lm.z;
    }
  }
  return out;
}
