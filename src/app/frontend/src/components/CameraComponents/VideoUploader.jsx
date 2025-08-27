import React, { useState } from "react";
import RoundedButton from "../RoundedButton";

const VideoUploader = ({ onNext, onBack }) => {
  const [selected, setSelected] = useState(null);

  const options = [
    { id: "raw", name: "Raw Dough" },
    { id: "baked", name: "Baked Baguette" },
  ];

  const handleSelect = (id) => {
    setSelected(id);
  };

  
  const handleDisconnect = () => {
    setSelected(null);
  };
  return (
      <div className="flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[106px] py-5 px-8">
      <p className="text-2xl">Select Video</p>

      <div className="flex gap-10">
        {options.map((option) => {
          const isSelected = selected === option.id;
          return (
            <div
              key={option.id}
              className={`group w-[335px] h-[160px] rounded-[12px] p-5 backdrop-blur-[110.57px] flex flex-col justify-between cursor-pointer transition-all
                ${
                  isSelected
                    ? "border-[#1272E5] bg-[rgba(0,0,0,0.39)] border-2"
                    : "border-[#0E2332] bg-[rgba(0,0,0,0.04)] hover:border-[#1272E5] border-2"
                }`}
              onClick={() => handleSelect(option.id)}
            >
               <div className="flex justify-between items-center">
              <p className="text-lg font-medium">{option.name}</p>
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
                  {/* <div
                    className={`text-[10px] font-semibold rounded-4xl py-1 px-5 ${
                      status === "Connected" ? "bg-[#1AA64B]" : "bg-[#FF4C40]"
                    }`}
                  >
                    {status}
                  </div> */}
                </div>
                 </div>
              <div className="mt-4">
                <RoundedButton label="Select" active={isSelected} />
              </div>
            </div>
          );
        })}
      </div>

 
      <div className="mt-6 flex justify-between">
        <button onClick={onBack} className="px-5 py-2 rounded-xl border cursor-pointer border-[#1272E5] text-[#1272E5]">
          Select Camera
        </button>
        <button
          onClick={onNext}
          disabled={!selected}
       className={`px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer disabled:opacity-50`}
        >
          Preview
        </button>
      </div>
    </div>
  );
};

export default VideoUploader;

