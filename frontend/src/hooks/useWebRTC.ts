import { useEffect, useRef, useState } from "react";
import type { Socket } from "socket.io-client";

// STUN handles ~80% of NATs (same-network, simple NATs).  Cross-network peers
// behind symmetric NATs (cellular carriers, hotel wifi, corporate firewalls)
// need a TURN relay to connect at all.  Multiple TURN endpoints are listed so
// the browser can fall back from UDP to TCP/443 on networks that block UDP.
//
// These credentials are tied to a free Metered.ca account (500MB/month relay
// quota — enough for demos but not production traffic).  For real production
// use, generate ephemeral tokens server-side instead of shipping a long-lived
// credential in the client bundle.
const METERED_USERNAME = "744e55666a98a008b171d14c";
const METERED_CREDENTIAL = "ay/FzqiyBEtuN+9O";

const ICE_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.relay.metered.ca:80" },
  { urls: "stun:stun.l.google.com:19302" },
  {
    urls: "turn:standard.relay.metered.ca:80",
    username: METERED_USERNAME,
    credential: METERED_CREDENTIAL,
  },
  {
    urls: "turn:standard.relay.metered.ca:80?transport=tcp",
    username: METERED_USERNAME,
    credential: METERED_CREDENTIAL,
  },
  {
    urls: "turn:standard.relay.metered.ca:443",
    username: METERED_USERNAME,
    credential: METERED_CREDENTIAL,
  },
  {
    urls: "turns:standard.relay.metered.ca:443?transport=tcp",
    username: METERED_USERNAME,
    credential: METERED_CREDENTIAL,
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
  const [peerReady, setPeerReady] = useState(false);

  // Build / tear down the peer connection alongside the socket.
  useEffect(() => {
    if (!socket || !localStream) return;

    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    pcRef.current = pc;

    localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

    // Use the remote stream from the ontrack event directly.  An empty
    // MediaStream + mutation pattern silently fails because React doesn't
    // notice the mutation, and <video srcObject> is only re-assigned on
    // reference change.  Promoting the event's stream into state forces the
    // RemoteVideo component to re-render with a stream that already has tracks.
    pc.ontrack = (ev) => {
      const stream = ev.streams[0];
      if (stream) setRemoteStream(stream);
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

    const handlePeerReady = () => setPeerReady(true);

    socket.on("webrtc_offer", handleOffer);
    socket.on("webrtc_answer", handleAnswer);
    socket.on("webrtc_ice", handleIce);
    socket.on("webrtc_ready", handlePeerReady);

    // Announce that our peer connection is built and listeners are wired.
    // The initiator uses this signal to (re-)send the offer, fixing the race
    // where the initiator emits before the answerer has a pc.
    socket.emit("webrtc_ready", {});

    return () => {
      socket.off("webrtc_offer", handleOffer);
      socket.off("webrtc_answer", handleAnswer);
      socket.off("webrtc_ice", handleIce);
      socket.off("webrtc_ready", handlePeerReady);
      pc.close();
      pcRef.current = null;
      setPeerReady(false);
    };
  }, [socket, localStream]);

  // Initiator sends an offer once the peer is both present (room_state) and
  // ready (their pc + listeners exist). We also re-offer if `peerReady` fires
  // again after a peer rejoin.
  useEffect(() => {
    const pc = pcRef.current;
    if (!pc || !socket || !isInitiator || !peerPresent || !peerReady) return;

    let cancelled = false;
    (async () => {
      try {
        // If we're mid-negotiation, roll back to stable before re-offering.
        if (pc.signalingState !== "stable") {
          await pc.setLocalDescription({ type: "rollback" }).catch(() => {});
        }
        const offer = await pc.createOffer();
        if (cancelled) return;
        await pc.setLocalDescription(offer);
        socket.emit("webrtc_offer", { sdp: offer });
      } catch (err) {
        console.warn("[useWebRTC] offer failed:", err);
      }
    })();

    return () => { cancelled = true; };
  }, [socket, isInitiator, peerPresent, peerReady]);

  return { remoteStream, state };
}
