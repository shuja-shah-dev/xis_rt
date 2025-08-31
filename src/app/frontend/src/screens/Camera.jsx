import React, { useState } from "react";
import CameraSelector from "../components/CameraComponents/CameraSelector";
import CTIUploader from "../components/CameraComponents/CTIUploader";
import VideoUploader from "../components/CameraComponents/VideoUploader";

const Camera = ({ setActiveScreen, setSelectedVideo  }) => {
  const [step, setStep] = useState(1);
  const [cameraType, setCameraType] = useState({ id: null, name: "" });

  return (
    <div>
      <p className="text-3xl font-semibold">Camera Initialization</p>
      <p className="text-sm text-[#73768D] font-roboto">
        Initialize camera quickly for seamless, accurate real-time inference
      </p>

      {/* progress indicator */}
      <div className="flex justify-center mt-6">
        <div className="flex items-center w-[40%]">
          {[1, 2].map((num, idx) => (
            <React.Fragment key={num}>
              <div className="flex flex-col items-center flex-1">
                <div
                  className={`flex items-center justify-center w-8 h-8 rounded-full border ${
                    step === num
                      ? "bg-white text-black border-black"
                      : "text-white border-white"
                  }`}
                >
                  {num}
                </div>
                <p
                  className={`text-sm mt-2 ${
                    step === num ? "" : "text-[#9A9A9A]"
                  }`}
                >
                  {num === 1
                    ? "Camera Selection"
                    : cameraType.id === "video"
                    ? "Select Video"
                    : "Upload CTI File"}
                </p>
              </div>
              {idx < 1 && <div className="flex-1 h-[1px] bg-[#C1C1C1]"></div>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* steps */}
      <div className="mt-12">
        {step === 1 && (
          <CameraSelector
            onNext={(id, name) => {
              setCameraType({ id, name }); 
              setStep(2);
            }}
          />
        )}

        {step === 2 &&
          (cameraType.id === "video" ? (
            <VideoUploader
              onNext={(videoId) => {        
              setSelectedVideo(videoId);  
              setActiveScreen("Model");
            }}
              onBack={() => setStep(1)}
            />
          ) : (
            <CTIUploader
              cameraName={cameraType.name}
               onNext={() => {
          setSelectedVideo(null);      
          setActiveScreen("Model");
        }}
              onBack={() => setStep(1)}
            />
          ))}
      </div>
    </div>
  );
};

export default Camera;
