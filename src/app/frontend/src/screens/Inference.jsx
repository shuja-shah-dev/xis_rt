import React, { useEffect, useState, useRef } from "react";
import { io } from "socket.io-client";

const Inference = ({ setActiveScreen, disconnectStream, setCameraStatus, socketRef }) => {
  const [streaming, setStreaming] = useState(false);
  const [socket, setSocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("Disconnected");
  const [disconnecting, setDisconnecting] = useState(false);
  const [videoEnded, setVideoEnded] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState(null);

  useEffect(() => {
    const savedSelected = localStorage.getItem("selectedCamera");
    if (savedSelected) {
      const { id, name } = JSON.parse(savedSelected);
      setSelectedCamera(name);
    }
  }, []);

  useEffect(() => {
    if (streaming && !socketRef.current) {
      const newSocket = io("http://localhost:5000/ws", {
        transports: ["websocket", "polling"],
        forceNew: true
      });

      newSocket.on("connect", () => {
        console.log("Connected to WebSocket /ws namespace");
        setConnectionStatus("Connected");
        newSocket.emit("join_stream");
      });

      newSocket.on("disconnect", () => {
        console.log("Disconnected from WebSocket");
        setConnectionStatus("Disconnected");
      });

      newSocket.on("connect_error", (error) => {
        console.error("Connection error:", error);
        setConnectionStatus("Connection Error");
      });

      newSocket.on("status", (data) => {
        console.log("Status update:", data);
      });

      newSocket.on('video_ended', (data) => {
        console.log('succesffully reached video_ended event')
        console.log(data);
        setVideoEnded(true);

      });

      newSocket.on("stream_frame", (data) => {
        console.log("=== STREAM FRAME RECEIVED ===");
        console.log("Frame data:", data.frame ? `Base64 string length: ${data.frame.length}` : "NO FRAME DATA");
        console.log("Metrics:", data.metrics);
        console.log("Full data object:", data);

        const img = document.getElementById("camera-stream");
        if (img && data.frame) {
          img.src = "data:image/jpeg;base64," + data.frame;
          console.log("Image src updated successfully");
        } else {
          console.log("Failed to update image - img element or frame data missing");
        }
      });

      newSocket.on("inference_result", (data) => {
        console.log("=== INFERENCE RESULT RECEIVED ===");
        console.log("Frame data:", data.frame ? `Base64 string length: ${data.frame.length}` : "NO FRAME DATA");
        console.log("Detections:", data.detections?.length || 0);
        console.log("Metrics:", data.metrics);
        console.log("Full data object:", data);

        const img = document.getElementById("camera-stream");
        if (img && data.frame) {
          img.src = "data:image/jpeg;base64," + data.frame;
          console.log("Image src updated successfully");
        } else {
          console.log("Failed to update image - img element or frame data missing");
        }
      });

      socketRef.current = newSocket;
      setSocket(newSocket);
    }

    return () => {
      if (socketRef.current && !streaming) {
        socketRef.current.disconnect();
        socketRef.current = null;
        setSocket(null);
        setConnectionStatus("Disconnected");
      }
    };
  }, [streaming]);


  const startStream = async () => {
    try {
      console.log("Btn pressed")

      const res = await fetch("http://localhost:5000/api/stream/normal", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        }
      });

      const data = await res.json();
      console.log("response", data)

      if (res.ok) {
        console.log("Stream started successfully:", data);
        setStreaming(true);
       
        setCameraStatus("Connected");
      } else {
        console.error("Failed to start stream:", data);
        alert(`Failed to start stream: ${data.message || 'Unknown error'}`);
      }
    } catch (err) {
      console.error("Error starting stream:", err);
      alert(`Error starting stream: ${err.message}`);
    }
  };

  const handleDisconnect = async () => {
    setDisconnecting(true);
    await disconnectStream();

    setConnectionStatus("Disconnected");
    setCameraStatus("Disconnected");
    setTimeout(() => {
      setDisconnecting(false);
      setStreaming(false);
      setActiveScreen("Camera");
    }, 2000);
  };

  return (
    <div>
      <p className="text-3xl font-semibold">Inference</p>
      <p className="text-sm text-[#73768D]">
        Run real-time predictions on feed using the selected trained model
      </p>

      <div className="relative h-[680px] border-2 border-[#0E2332] bg-[rgba(0,0,0,0.11)] backdrop-blur-[106px] rounded-xl p-4 flex items-center justify-center mt-6">
        <div className="relative w-full h-full flex items-center justify-center">
          <img
            id="camera-stream"
            alt="Inference"
            className={`rounded-xl transition-all duration-300 ${streaming
              ? "object-none w-full h-full"
              : "w-10 h-10 object-contain"
              }`}
            src={streaming ? undefined : "/play.svg"}
          />
          {!streaming && (
            <>
              <p className="text-lg absolute bottom-16 w-full text-center">Inference Display</p>
              <p className="text-sm text-[#73768D] absolute bottom-10 w-full text-center">
                Initialize Camera and Load Model to Start
              </p>
            </>
          )}
        </div>

      </div>

      <div className="mt-6 flex gap-4 justify-end">


        {!streaming ? (
          <button
            onClick={startStream}
            className="px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer"
          >
            Start Inference
          </button>
        ) : (
          <button
            onClick={handleDisconnect}
            disabled={
              disconnecting ||
              (selectedCamera === "Video As Webcam" && !videoEnded)
            }
            className={`px-5 py-2 rounded-xl text-white text-md transition-colors ${disconnecting || (selectedCamera === "Video As Webcam" && !videoEnded)
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-[#dc2626] hover:bg-red-700 cursor-pointer"
              }`}
            title={
              disconnecting
                ? "Currently disconnecting..."
                : selectedCamera === "Video As Webcam" && !videoEnded
                  ? "Please let inference finish before disconnecting"
                  : ""
            }
          >
            {disconnecting ? "Disconnecting..." : "Disconnect"}
          </button>
        )}

      </div>

      {streaming && (
        <div className="mt-4 p-4 bg-black rounded-lg" >
          <p className="text-sm text-white">
            Streaming: <span className="font-semibold">{streaming ? "Active" : "Inactive"}</span>
          </p>
          <p className="text-sm text-white">
            WebSocket: <span className="font-semibold">{connectionStatus}</span>
          </p>
        </div>
      )}
    </div>
  );
};

export default Inference;