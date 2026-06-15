"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface RemoteVideoProps {
  stream: MediaStream | null;
  muted?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

export function RemoteVideo({ stream, muted = false, style, className }: RemoteVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  // iOS Safari and some Android browsers reject autoplay when the stream has
  // unmuted audio.  Surface a tap-to-play affordance instead of silently
  // showing a black tile.
  const [needsTap, setNeedsTap] = useState(false);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || !stream) return;
    if (el.srcObject !== stream) {
      el.srcObject = stream;
    }
    el.play()
      .then(() => setNeedsTap(false))
      .catch(() => setNeedsTap(true));
  }, [stream]);

  const handleTap = () => {
    const el = videoRef.current;
    if (!el) return;
    el.play()
      .then(() => setNeedsTap(false))
      .catch(() => {});
  };

  return (
    <div className={cn("relative block w-full bg-black", className)} style={style}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted={muted}
        className="block h-full w-full object-cover"
      />
      {needsTap && (
        <button
          type="button"
          onClick={handleTap}
          aria-label="Tap to play remote video"
          className="absolute inset-0 flex items-center justify-center bg-black/70 text-white"
        >
          <span className="flex flex-col items-center gap-2">
            <span aria-hidden className="text-4xl">▶</span>
            <span className="text-sm">Tap to start video</span>
          </span>
        </button>
      )}
      {!stream && !needsTap && (
        // No stream yet — distinguishes "still connecting" from "video failed
        // to autoplay" so the peer tile isn't a silent black box for 2-3s
        // during the WebRTC handshake.
        <div
          aria-live="polite"
          className="pointer-events-none absolute inset-0 flex items-center justify-center text-xs text-white/70"
        >
          Connecting…
        </div>
      )}
    </div>
  );
}
