import React, { useState } from "react";
import RoundedButton from "../RoundedButton";

const CameraSelector = ({ onNext }) => {
  const [selected, setSelected] = useState(null);

  const cameras = [
    { id: "video", name: "Video As Webcam" },
    { id: "allied", name: "Allied Camera" },
    { id: "ids", name: "IDS Camera" },
  ];

  const handleSelect = (id) => {
    setSelected(id);
  };

  const handleDisconnect = () => {
    setSelected(null);
  };

  return (
    <div className="flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[106.0999984741211px] py-5 px-8">
      <p className="text-2xl">Choose Your Camera</p>

      <div className="flex gap-10">
        {cameras.map((camera) => {
          const isSelected = selected === camera.id;
          const status = isSelected ? "Connected" : "Not Connected";

          return (
            <div
              key={camera.id}
              className={`group w-[335px] h-[160px]  rounded-[12px] p-5 backdrop-blur-[110.57px] transition 
                ${
                  isSelected
                    ? "border-[#1272E5] bg-[rgba(0,0,0,0.39)] border"
                    : "border-[#0E2332] bg-[rgba(0,0,0,0.04)] hover:border-[#1272E5] border-2"
                }`}
            >
              
              <div className="flex justify-between items-center">
                <p className="text-lg font-medium">{camera.name}</p>

                <div className="flex items-center gap-2">
                  {isSelected && (
                    <div className="cursor-pointer" onClick={handleDisconnect}>
                      <img
                        src="/disconnector.svg"
                        alt="Disconnector"
                        width={21}
                        height={21}
                      />
                    </div>
                  )}
                  <div
                    className={`text-[10px] font-semibold rounded-4xl py-1 px-5 ${
                      status === "Connected" ? "bg-[#1AA64B]" : "bg-[#FF4C40]"
                    }`}
                  >
                    {status}
                  </div>
                </div>
              </div>

              <div className="mt-2 h-[0.5px] bg-[linear-gradient(90deg,rgba(255,255,255,0)_0%,rgba(255,255,255,0.31)_52.4%,rgba(255,255,255,0)_100%)]"></div>

              <div className="mt-6">
                <RoundedButton
                  label="Select"
                  active={isSelected}
                  onClick={() => handleSelect(camera.id)}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 flex justify-end">
        <button
          disabled={!selected}
          onClick={onNext}
          className={`px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer disabled:opacity-50`}
        >
          Upload CTI File
        </button>
      </div>
    </div>
  );
};

export default CameraSelector;