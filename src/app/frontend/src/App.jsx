import { useState } from "react";
import Header from "./components/Header";
import Switcher from "./components/Switcher";

import Camera from "./screens/Camera";
import Model from "./screens/Model";
import Inference from "./screens/Inference";
import { BlobLarge } from "./components/Blob";
import ImageMeasurementTool from "./components/Measurement";
import Measure from "./components/Measure";

function App() {
  const [activeScreen, setActiveScreen] = useState("Camera");

  const renderScreen = () => {
    switch (activeScreen) {
      case "Camera":
        return <Camera setActiveScreen={setActiveScreen} />;
      case "Model":
        return <Model setActiveScreen={setActiveScreen} />;
      case "Inference":
        return <Inference />;
      default:
        return null;
    }
  };

  return (
    <div className="relative">
       <BlobLarge className="top-[-200px] left-[-400px]" />
       <ImageMeasurementTool/>
       <Measure/>
      <Header />
      <Switcher activeScreen={activeScreen} onChange={setActiveScreen} />
      <div className="mt-6">{renderScreen()}</div>
    </div>
  );
}

export default App;
