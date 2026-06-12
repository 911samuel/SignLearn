"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface RemoteVideoProps {
  stream: MediaStream | null;
  muted?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

export function RemoteVideo({ stream, muted = false, style, className }: RemoteVideoProps) {
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
      className={cn("block w-full bg-black", className)}
      style={style}
    />
  );
}
