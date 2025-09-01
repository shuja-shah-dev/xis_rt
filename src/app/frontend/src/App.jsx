import { useState, useEffect, useRef } from "react";
import Header from "./components/Header";
import Switcher from "./components/Switcher";

import Camera from "./screens/Camera";
import Model from "./screens/Model";
import Inference from "./screens/Inference";
import { BlobLarge } from "./components/Blob";

function App() {
  const [activeScreen, setActiveScreen] = useState("Camera");
  const [selectedVideo, setSelectedVideo] = useState(null);
  const socketRef = useRef(null);
  const prevScreenRef = useRef(null);

  const [modelStatus, setModelStatus] = useState("Not Loaded");
  const [mqttStatus, setMqttStatus] = useState("Disconnected");
  const [cameraStatus, setCameraStatus] = useState("Disconnected");
  
   const [selectedCustomCam, setSelectedCustomCam] = useState(null);

  // useEffect(() => {
  //   const stored = localStorage.getItem("selectedCamera");
   
  //   if (stored) {
  //     setSavedSelected(JSON.parse(stored));
  //   }
  // }, []);

  // useEffect(() => {
   
  //   if (selectedCustomCam ) {
      
  //     setCameraStatus("Connected");

  //   } else {
  //     setCameraStatus("Disconnected");
  //   }
  // }, [selectedCustomCam]);
  useEffect(() => {
    const checkMqtt = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/status");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        let messages = data.messages || [];

        if (messages) {
          setMqttStatus("Connected");
        } else {
          setMqttStatus("Disconnected");
        }
       
      } catch (e){
        setMqttStatus("Disconnected");
      
      }
    };
    checkMqtt();
    const interval = setInterval(checkMqtt, 5000);
    return () => clearInterval(interval);
  }, []);

  const disconnectStream = async () => {
    try {
      await fetch("http://localhost:5000/api/stream/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }).catch(() => { });

      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
      // setCameraStatus("Disconnected");
      console.log("Stream disconnected globally");
    } catch (err) {
      console.error("Global disconnect error:", err);
    }
  };

  useEffect(() => {
    if (prevScreenRef.current === "Inference" && activeScreen !== "Inference") {
      disconnectStream();
    }
    prevScreenRef.current = activeScreen;
  }, [activeScreen]);

  useEffect(() => {
    const handleBeforeUnload = () => disconnectStream();
    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      handleBeforeUnload();
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);


  const renderScreen = () => {
    switch (activeScreen) {
      case "Camera":
        return <Camera setActiveScreen={setActiveScreen} setSelectedVideo={setSelectedVideo} setSelectedCustomCam={setSelectedCustomCam}/>;
      case "Model":
        return <Model setActiveScreen={setActiveScreen} selectedVideo={selectedVideo} setModelStatus={setModelStatus} />;
      case "Inference":
        return <Inference setActiveScreen={setActiveScreen}
          disconnectStream={disconnectStream}
          setCameraStatus={setCameraStatus}
          socketRef={socketRef} />;
      default:
        return null;
    }
  };

  return (
    <div className="relative">
      <BlobLarge className="top-[-200px] left-[-400px]" />
      <Header modelStatus={modelStatus}
        mqttStatus={mqttStatus}
        cameraStatus={cameraStatus} />
      <Switcher activeScreen={activeScreen} onChange={setActiveScreen} />
      <div className="mt-6">{renderScreen()}</div>
    </div>
  );
}

export default App;
