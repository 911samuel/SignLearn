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

  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const pausedRef = useRef(false);
  const lastFrameTsRef = useRef<number>(0);

  // Initialise MediaPipe HandLandmarker
  useEffect(() => {
    let cancelled = false;

    async function init() {
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
    }

    init();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  // Listen for prediction events on the shared socket
  useEffect(() => {
    if (!socket) return;
    const handler = (data: Prediction) => {
      if (data.ready && lastFrameTsRef.current) {
        setLatencyMs(Date.now() - lastFrameTsRef.current);
      }
      setPrediction(data);
    };
    socket.on("prediction", handler);
    return () => {
      socket.off("prediction", handler);
    };
  }, [socket]);

  // rAF loop: extract landmarks → emit
  useEffect(() => {
    function tick() {
      rafRef.current = requestAnimationFrame(tick);
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;

      if (!video || !landmarker || !socket || video.readyState < 2) return;
      if (video.currentTime === lastVideoTimeRef.current) return;
      lastVideoTimeRef.current = video.currentTime;

      const result: HandLandmarkerResult = landmarker.detectForVideo(
        video,
        performance.now()
      );

      setLandmarkerResult(result);
      if (!pausedRef.current) {
        const landmarks = flattenLandmarks(result);
        lastFrameTsRef.current = Date.now();
        socket.emit("frame", { landmarks, t: lastFrameTsRef.current });
      }
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [videoRef, socket]);

  const reset = useCallback(() => {
    socket?.emit("reset");
    setPrediction({ label: null, confidence: null, ready: false });
  }, [socket]);

  const togglePaused = useCallback(() => {
    setPaused((p) => {
      pausedRef.current = !p;
      return !p;
    });
  }, []);

  return { prediction, landmarkerResult, reset, paused, togglePaused, latencyMs };
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
