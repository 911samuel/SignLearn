import { useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";

// STUN only works for ~80% of NATs (same-network, simple NATs).  Cross-network
// peers behind symmetric NATs or strict firewalls need a TURN server to relay
// the video stream.  Open Relay Project provides free TURN service that is
// sufficient for demos; for production, switch to a paid provider (Xirsys,
// Twilio NAT Traversal, or self-hosted coturn) with auth tokens.
const ICE_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
  {
    urls: "turn:openrelay.metered.ca:80",
    username: "openrelayproject",
    credential: "openrelayproject",
  },
  {
    urls: "turn:openrelay.metered.ca:443",
    username: "openrelayproject",
    credential: "openrelayproject",
  },
  {
    urls: "turn:openrelay.metered.ca:443?transport=tcp",
    username: "openrelayproject",
    credential: "openrelayproject",
  },
];

export type IceState = RTCPeerConnectionState;

/**
 * Peer-to-peer audio+video using WebRTC, signaled over the room socket.
 *
 * The Signer is the "polite" peer that creates the initial offer once a
 * second participant joins (i.e. when `peerPresent` flips to true).
 */
export function useWebRTC(
  socket: Socket | null,
  localStream: MediaStream | null,
  isInitiator: boolean,
  peerPresent: boolean,
) {
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [state, setState] = useState<IceState>("new");

  // Build / tear down the peer connection alongside the socket.
  useEffect(() => {
    if (!socket || !localStream) return;

    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    pcRef.current = pc;
    const remote = new MediaStream();
    setRemoteStream(remote);

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    pc.ontrack = (ev) => {
      ev.streams[0].getTracks().forEach((t) => {
        if (!remote.getTracks().includes(t)) remote.addTrack(t);
      });
    };

    pc.onicecandidate = (ev) => {
      if (ev.candidate) socket.emit("webrtc_ice", { candidate: ev.candidate.toJSON() });
    };

    pc.onconnectionstatechange = () => setState(pc.connectionState);

    const handleOffer = async (data: { sdp: RTCSessionDescriptionInit }) => {
      await pc.setRemoteDescription(data.sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit("webrtc_answer", { sdp: answer });
    };

    const handleAnswer = async (data: { sdp: RTCSessionDescriptionInit }) => {
      if (pc.signalingState === "have-local-offer") {
        await pc.setRemoteDescription(data.sdp);
      }
    };

    const handleIce = async (data: { candidate: RTCIceCandidateInit }) => {
      try {
        await pc.addIceCandidate(data.candidate);
      } catch (err) {
        console.warn("[useWebRTC] addIceCandidate failed:", err);
      }
    };

    socket.on("webrtc_offer", handleOffer);
    socket.on("webrtc_answer", handleAnswer);
    socket.on("webrtc_ice", handleIce);

    return () => {
      socket.off("webrtc_offer", handleOffer);
      socket.off("webrtc_answer", handleAnswer);
      socket.off("webrtc_ice", handleIce);
      pc.close();
      pcRef.current = null;
    };
  }, [socket, localStream]);

  // When the peer arrives (or we're the initiator and they're already present),
  // send an offer.
  useEffect(() => {
    const pc = pcRef.current;
    if (!pc || !socket || !isInitiator || !peerPresent) return;
    if (pc.signalingState !== "stable") return;

    let cancelled = false;
    (async () => {
      const offer = await pc.createOffer();
      if (cancelled) return;
      await pc.setLocalDescription(offer);
      socket.emit("webrtc_offer", { sdp: offer });
    })();

    return () => { cancelled = true; };
  }, [socket, isInitiator, peerPresent]);

  return { remoteStream, state };
}
