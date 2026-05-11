"use client";

import { useEffect, useRef } from "react";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

interface Props {
  result: HandLandmarkerResult | null;
  width: number;
  height: number;
}

const CONNECTIONS: [number, number][] = [
  [0, 1],[1, 2],[2, 3],[3, 4],
  [0, 5],[5, 6],[6, 7],[7, 8],
  [0, 9],[9, 10],[10, 11],[11, 12],
  [0, 13],[13, 14],[14, 15],[15, 16],
  [0, 17],[17, 18],[18, 19],[19, 20],
  [5, 9],[9, 13],[13, 17],
];

// Lerp factor per frame — higher = snappier, lower = smoother.
const LERP = 0.45;

type XYZ = { x: number; y: number; z: number };

function lerp1(a: number, b: number, t: number) { return a + (b - a) * t; }
function lerpPt(a: XYZ, b: XYZ, t: number): XYZ {
  return { x: lerp1(a.x, b.x, t), y: lerp1(a.y, b.y, t), z: lerp1(a.z, b.z, t) };
}

export function LandmarkOverlay({ result, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prevLandmarks = useRef<XYZ[][] | null>(null);
  const reduceMotion = useRef(
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    if (!result || result.landmarks.length === 0) {
      prevLandmarks.current = null;
      return;
    }

    const current = result.landmarks as XYZ[][];

    // Interpolate toward current from previous when motion is allowed.
    let drawn: XYZ[][];
    if (!reduceMotion.current && prevLandmarks.current && prevLandmarks.current.length === current.length) {
      drawn = current.map((hand, hi) =>
        hand.map((pt, pi) => lerpPt(prevLandmarks.current![hi][pi], pt, LERP))
      );
    } else {
      drawn = current;
    }
    prevLandmarks.current = drawn;

    for (const hand of drawn) {
      // Skeleton connections
      ctx.strokeStyle = "rgba(0, 229, 255, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.lineCap = "round";
      for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * width, hand[a].y * height);
        ctx.lineTo(hand[b].x * width, hand[b].y * height);
        ctx.stroke();
      }

      // Knuckle / palm dots slightly larger, fingertip dots bright
      for (let i = 0; i < hand.length; i++) {
        const lm = hand[i];
        const isTip = [4, 8, 12, 16, 20].includes(i);
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, isTip ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = isTip ? "rgba(255, 255, 255, 0.95)" : "rgba(0, 229, 255, 0.85)";
        ctx.fill();
      }
    }
  }, [result, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      aria-hidden="true"
      style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none", width: "100%", height: "100%" }}
    />
  );
}
