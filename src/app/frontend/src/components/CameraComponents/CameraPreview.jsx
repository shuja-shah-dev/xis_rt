import React from "react";

const CameraPreview = ({ onNext, onBack }) => {
  return (
    <div className="flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[106.0999984741211px] py-5 px-8">
      <p className="text-2xl">Camera Preview</p>

      <div className="border rounded-xl border-[rgba(255,255,255,0.4)] h-[244px] p-4 flex items-center justify-center">
        <div className="m-auto w-[500px] text-center">
          <img src="/camera.svg" alt="preview camera" className="m-auto" width={68} height={68} />
          <p className="text-lg">Preview</p>
        </div>
      </div>

      <div className="mt-6 flex justify-between">
        <button onClick={onBack} className="px-5 py-2 rounded-xl border border-[#1272E5] text-[#1272E5]">
          Upload CTI File
        </button>
        <button
          onClick={onNext}
         
          className={`px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer`}
        >
          Configure Model
        </button>
      </div>
    </div>
  );
};

export default CameraPreview;
