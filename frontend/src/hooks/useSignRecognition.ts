import { useEffect, useRef, useCallback, useState } from "react";
import {
  HandLandmarker,
  FilesetResolver,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { io, type Socket } from "socket.io-client";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:5001";
const FEATURE_DIM = 126; // 2 hands × 21 landmarks × 3 coords

export interface Prediction {
  label: string | null;
  confidence: number | null;
  ready: boolean;
}

export function useSignRecognition(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [prediction, setPrediction] = useState<Prediction>({
    label: null,
    confidence: null,
    ready: false,
  });
  const [connected, setConnected] = useState(false);
  const [landmarkerResult, setLandmarkerResult] = useState<HandLandmarkerResult | null>(null);

  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);

  // Initialise MediaPipe HandLandmarker (WASM, runs in-browser)
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

  // WebSocket connection
  useEffect(() => {
    const socket = io(BACKEND_URL, { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));
    socket.on("prediction", (data: Prediction) => setPrediction(data));

    return () => {
      socket.disconnect();
    };
  }, []);

  // rAF loop: extract landmarks → emit frame
  useEffect(() => {
    function tick() {
      rafRef.current = requestAnimationFrame(tick);
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      const socket = socketRef.current;

      if (!video || !landmarker || !socket || video.readyState < 2) return;
      if (video.currentTime === lastVideoTimeRef.current) return;
      lastVideoTimeRef.current = video.currentTime;

      const result: HandLandmarkerResult = landmarker.detectForVideo(
        video,
        performance.now()
      );

      setLandmarkerResult(result);
      const landmarks = flattenLandmarks(result);
      socket.emit("frame", { landmarks, t: Date.now() });
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [videoRef]);

  const reset = useCallback(() => {
    socketRef.current?.emit("reset");
    setPrediction({ label: null, confidence: null, ready: false });
  }, []);

  return { prediction, connected, landmarkerResult, reset };
}

// Flatten up to 2 hands into a fixed-length 126-float array.
// Hand 0 → indices 0–62, hand 1 → indices 63–125. Missing hand → zeros.
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
