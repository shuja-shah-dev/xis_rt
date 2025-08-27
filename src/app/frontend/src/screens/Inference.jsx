import React, { useEffect, useState } from "react";
import { io } from "socket.io-client";

const Inference = () => {
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    let socket;

    if (streaming) {
      // Connect to WebSocket server
      socket = io("http://localhost:5000/ws");

      socket.on("connect", () => {
        console.log("Connected to WebSocket");
      });

      socket.on("disconnect", () => {
        console.log("Disconnected from WebSocket");
      });

      // Listen for stream frames
      socket.on("stream_frame", (data) => {
        console.log("frame streaming")
        const img = document.getElementById("camera-stream");
        if (img && data.frame) {
          img.src = "data:image/jpeg;base64," + data.frame;
          console.log("frame updated")
        }

        // Show metrics in console
        if (data.metrics) {
          console.log("FPS:", data.metrics.display_fps);
        }
      });

      return () => {
        socket.disconnect();
      };
    }
  }, [streaming]);

  const startStream = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/stream/", {
        method: "POST",
      });

      if (res.ok) {
        console.log("Stream started successfully");
        setStreaming(true);
      } else {
        console.error("Failed to start stream");
      }
    } catch (err) {
      console.error("Error starting stream:", err);
    }
  };

  return (
    <div>
      <p className="text-3xl font-semibold">Inference</p>
      <p className="text-sm text-[#73768D]">
        Run real-time predictions on feed using the selected trained model
      </p>

    <div className="relative h-[450px] border-2 border-[#0E2332] bg-[rgba(0,0,0,0.11)] backdrop-blur-[106px] rounded-xl p-4 flex items-center justify-center mt-6">
  <div className="m-auto w-full h-full flex flex-col items-center justify-center text-center">
    <img
      id="camera-stream"
      src="/play.svg" // placeholder before streaming starts
      alt="Inference"
      className="object-contain w-full h-full rounded-xl"
    />
    {!streaming && (
      <>
        <p className="text-lg absolute bottom-16 w-full">Inference Display</p>
        <p className="text-sm text-[#73768D] absolute bottom-10 w-full">
          Initialize Camera and Load Model to Start
        </p>
      </>
    )}
  </div>
</div>


         {/* <div className="relative h-[450px] border-2 border-[#0E2332] bg-[rgba(0,0,0,0.11)] backdrop-blur-[106.0999984741211px] rounded-xl p-4 flex items-center justify-center mt-6">
       
        <div className="m-auto w-[500px] text-center">
          <img
            src="/play.svg"
            alt="Inference"
            className="m-auto"
            width={25}
            height={30}
          />
          <p className="text-lg">Inference Display</p>
          <p className="text-sm text-[#73768D]">
            Initialize Camera and Load Model to Start
          </p>
        </div>
      </div> */}

      <div className="mt-6 flex gap-4 justify-end">
        <button className="px-5 py-2 rounded-xl bg-[#4061A9]">Snapshot</button>
        <button
          onClick={startStream}
          className="px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer"
        >
          Start Inference
        </button>
      </div>
    </div>
  );
};

export default Inference;
