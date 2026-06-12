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

  // Listen for word_prediction events; flip the capture state machine back to idle.
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
    };
    socket.on("word_prediction", wordHandler);
    return () => {
      socket.off("word_prediction", wordHandler);
    };
  }, [socket]);

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
  }, [videoRef]);

  const reset = useCallback(() => {
    wordBufferRef.current = [];
    idleFramesRef.current = 0;
    prevLandmarksRef.current = null;
    stateRef.current = "idle";
    setCaptureStatus("idle");
    setCaptureProgress(0);
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
