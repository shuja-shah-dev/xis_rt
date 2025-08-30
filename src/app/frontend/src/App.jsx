import { useState , useEffect, useRef } from "react";
import Header from "./components/Header";
import Switcher from "./components/Switcher";

import Camera from "./screens/Camera";
import Model from "./screens/Model";
import Inference from "./screens/Inference";
import { BlobLarge } from "./components/Blob";

function App() {
  const [activeScreen, setActiveScreen] = useState("Camera");
   const socketRef = useRef(null); 
  const prevScreenRef = useRef(null);
  
  const disconnectStream = async () => {
    try {
      await fetch("http://localhost:5000/api/stream/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }).catch(() => {}); 

      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }

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
        return <Camera setActiveScreen={setActiveScreen} />;
      case "Model":
        return <Model setActiveScreen={setActiveScreen} />;
      case "Inference":
        return <Inference  setActiveScreen={setActiveScreen}
            disconnectStream={disconnectStream}
            socketRef={socketRef} />;
      default:
        return null;
    }
  };

  return (
    <div className="relative">
       <BlobLarge className="top-[-200px] left-[-400px]" />
      <Header />
      <Switcher activeScreen={activeScreen} onChange={setActiveScreen} />
      <div className="mt-6">{renderScreen()}</div>
    </div>
  );
}

export default App;
