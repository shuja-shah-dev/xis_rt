import React, { useEffect, useState, useRef } from "react";
import { io } from "socket.io-client";

const Inference = ({ setActiveScreen, disconnectStream, setCameraStatus, socketRef }) => {
  const [streaming, setStreaming] = useState(false);
  const [socket, setSocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("Disconnected");
  const [disconnecting, setDisconnecting] = useState(false);
  const [videoEnded, setVideoEnded] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [selectedMeasurements, setSelectedMeasurements] = useState([]);

  useEffect(() => {
    const savedSelected = localStorage.getItem("selectedCamera");
    if (savedSelected) {
      const { id, name } = JSON.parse(savedSelected);
      setSelectedCamera(name);
    }
  }, []);

  const [videoMode, setVideoMode] = useState(null);
  const [restarting, setRestarting] = useState(false);


  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/config");
        const data = await res.json();
        if (res.ok) {
          setVideoMode(data.video_mode);
        }
      } catch (err) {
        console.error("Error fetching config:", err);
      }
    };

    fetchConfig();
  }, []);


  useEffect(() => {
    if (streaming && !socketRef.current) {
      const newSocket = io("http://localhost:5000/ws", {
        transports: ["websocket", "polling"],
        forceNew: true,
      });

      newSocket.on("connect", () => {
        setConnectionStatus("Connected");
        newSocket.emit("join_stream");
      });

      newSocket.on("disconnect", () => {
        setConnectionStatus("Disconnected");
      });

      newSocket.on("connect_error", () => {
        setConnectionStatus("Connection Error");
      });

      newSocket.on("video_ended", () => {
        setVideoEnded(true);
      });

      newSocket.on("stream_frame", (data) => {
        const img = document.getElementById("camera-stream");
        if (img && data.frame) {
          img.src = "data:image/jpeg;base64," + data.frame;
        }
      });

      newSocket.on("inference_result", (data) => {
        const img = document.getElementById("camera-stream");
        if (img && data.frame) {
          img.src = "data:image/jpeg;base64," + data.frame;
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
      const res = await fetch("http://localhost:5000/api/stream/normal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          measurement_mode: selectedMeasurements,
        }),
      });

      const data = await res.json();
      if (res.ok) {
        setStreaming(true);
        setCameraStatus("Connected");
      } else {
        alert(`Failed to start stream: ${data.message || "Unknown error"}`);
      }
    } catch (err) {
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
    }, 3000);
  };


  const handleRestart = async () => {
    setRestarting(true);
    await disconnectStream();
    setStreaming(false);
    setConnectionStatus("Disconnected");
    setCameraStatus("Disconnected");

    setTimeout(async () => {
      await startStream();
      setRestarting(false);
    }, 3000);
  };

  const toggleMeasurement = (measurement) => {
    setSelectedMeasurements((prev) =>
      prev.includes(measurement)
        ? prev.filter((m) => m !== measurement)
        : [...prev, measurement]
    );
  };

  return (
    <div>
      <p className="text-3xl font-semibold">Inference</p>
      <p className="text-sm text-[#73768D]">
        Run real-time predictions on feed using the selected trained model
      </p>

      {/* Camera Stream */}
      <div className="relative h-[680px] border-2 border-[#0E2332] bg-[rgba(0,0,0,0.11)] rounded-xl p-4 flex items-center justify-center mt-6">
        <div className="relative w-full h-full flex items-center justify-center">
          <img
            id="camera-stream"
            alt="Inference"
            className={`rounded-xl transition-all duration-300 ${streaming ? "object-none w-full h-full" : "w-10 h-10 object-contain"
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
          <>
            <button
              onClick={startStream}
              className="px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer"
            >
              Start Inference
            </button>
          </>
        ) : (
          <button
            onClick={handleDisconnect}
            disabled={
              disconnecting ||
              (
                videoMode !== "donut" &&
                videoMode !== "baked_baguette" &&
                selectedCamera === "Video As Webcam" &&
                !videoEnded
              )
            }
            className={`px-5 py-2 rounded-xl text-white text-md transition-colors ${disconnecting ||
                (
                  videoMode !== "donut" &&
                  videoMode !== "baked_baguette" &&
                  selectedCamera === "Video As Webcam" &&
                  !videoEnded
                )
                ? "bg-gray-500 cursor-not-allowed"
                : "bg-[#dc2626] hover:bg-red-700 cursor-pointer"
              }`}
          >
            {disconnecting ? "Disconnecting..." : "Disconnect"}
          </button>

        )}

        <button
          onClick={handleRestart}
          disabled={restarting}
          className={`px-5 py-2 rounded-xl text-white text-md cursor-pointer transition-colors ${restarting ? "bg-gray-400 cursor-not-allowed" : "bg-[#f59e0b] hover:bg-yellow-600"
            }`}
        >
          {restarting ? "Restarting..." : "Restart"}
        </button>

      </div>

      <div className="mt-6 flex justify-end gap-4">
        {[
          { name: "Outer Diameter", mode: "donut" },
          { name: "Inner Diameter", mode: "donut" },
          { name: "Width", mode: "baked_baguette" },
          { name: "Height", mode: "baked_baguette" },
        ].map((btn) => {
          const enabled = videoMode === btn.mode;

          return (
            <button
              key={btn.name}
              disabled={!enabled}
              onClick={() => enabled && toggleMeasurement(btn.name)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors  ${!enabled
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : selectedMeasurements.includes(btn.name)
                  ? "bg-[#1AA64B] text-white cursor-pointer"
                  : "bg-white border border-gray-400 text-black hover:bg-gray-100 cursor-pointer"
                }`}
            >
              {btn.name}
            </button>
          );
        })}
      </div>

      {/* Status */}
      {streaming && (
        <div className="mt-4 p-4 bg-black rounded-lg">
          <p className="text-sm text-white">
            Streaming: <span className="font-semibold">{streaming ? "Active" : "Inactive"}</span>
          </p>
          <p className="text-sm text-white">
            WebSocket: <span className="font-semibold">{connectionStatus}</span>
          </p>
          <p className="text-sm text-white">Selected: {selectedMeasurements.join(", ") || "None"}</p>
        </div>
      )}
    </div>
  );
};

export default Inference;
