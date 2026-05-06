import { useEffect, useRef } from "react";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

interface Props {
  result: HandLandmarkerResult | null;
  width: number;
  height: number;
}

// Finger connections for drawing the skeleton
const CONNECTIONS: [number, number][] = [
  [0, 1],[1, 2],[2, 3],[3, 4],
  [0, 5],[5, 6],[6, 7],[7, 8],
  [0, 9],[9, 10],[10, 11],[11, 12],
  [0, 13],[13, 14],[14, 15],[15, 16],
  [0, 17],[17, 18],[18, 19],[19, 20],
  [5, 9],[9, 13],[13, 17],
];

export function LandmarkOverlay({ result, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    if (!result) return;

    for (const hand of result.landmarks) {
      // Draw connections
      ctx.strokeStyle = "rgba(0, 200, 255, 0.8)";
      ctx.lineWidth = 2;
      for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * width, hand[a].y * height);
        ctx.lineTo(hand[b].x * width, hand[b].y * height);
        ctx.stroke();
      }
      // Draw landmark dots
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      for (const lm of hand) {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [result, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
    />
  );
}
