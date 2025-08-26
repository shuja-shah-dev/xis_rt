import React, { useState } from "react";
import CameraSelector from "../components/CameraComponents/CameraSelector";
import CameraPreview from "../components/CameraComponents/CameraPreview";
import CTIUploader from "../components/CameraComponents/CTIUploader";
import VideoUploader from "../components/CameraComponents/VideoUploader"; // <-- new

const Camera = ({ setActiveScreen }) => {
  const [step, setStep] = useState(1);
  const [cameraType, setCameraType] = useState(null); // store selected

  return (
    <div>
      <p className="text-3xl font-semibold">Camera Initialization</p>
      <p className="text-sm text-[#73768D] font-roboto">
        Initialize camera quickly for seamless, accurate real-time inference
      </p>

      {/* progress indicator */}
      <div className="flex justify-center mt-6">
        <div className="flex items-center w-[40%]">
          {[1, 2, 3].map((num, idx) => (
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
                <p className={`text-sm mt-2 ${step === num ? "" : "text-[#9A9A9A]"}`}>
                  {num === 1
                    ? "Camera Selection"
                    : num === 2
                    ? cameraType === "video"
                      ? "Upload Video"
                      : "Upload CTI File"
                    : "Preview"}
                </p>
              </div>
              {idx < 2 && <div className="flex-1 h-[1px] bg-[#C1C1C1]"></div>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* steps */}
      <div className="mt-12">
        {step === 1 && (
          <CameraSelector
            onNext={(selected) => {
              setCameraType(selected);
              setStep(2);
            }}
          />
        )}

        {step === 2 &&
          (cameraType === "video" ? (
            <VideoUploader onNext={() => setStep(3)} onBack={() => setStep(1)} />
          ) : (
            <CTIUploader onNext={() => setStep(3)} onBack={() => setStep(1)} />
          ))}

        {step === 3 && (
          <CameraPreview
            onNext={() => setActiveScreen("Model")}
            onBack={() => setStep(2)}
          />
        )}
      </div>
    </div>
  );
};

export default Camera;
