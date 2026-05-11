"use client";

import { useEffect, useRef } from "react";

interface RemoteVideoProps {
  stream: MediaStream | null;
  muted?: boolean;
  style?: React.CSSProperties;
}

export function RemoteVideo({ stream, muted = false, style }: RemoteVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    if (el.srcObject !== stream) {
      el.srcObject = stream;
      el.play().catch(() => {});
    }
  }, [stream]);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted={muted}
      style={{ width: "100%", borderRadius: 8, background: "#000", ...style }}
    />
  );
}
