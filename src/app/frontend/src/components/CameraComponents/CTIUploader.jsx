import React, { useState } from "react";

const CTIUploader = ({ onNext, onBack, cameraName }) => {
  const [file, setFile] = useState(null);

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleChooseAgain = () => setFile(null);

  return (
    <div className="flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[106.0999984741211px] py-5 px-8">
      <p className="text-2xl">Upload {cameraName} CTI File</p>

      {!file ? (
        <div
          className="border rounded-lg border-dashed border-[rgba(255,255,255,0.4)] h-[244px] p-4 flex items-center justify-center"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <div className="m-auto w-[500px] text-center">
            <img src="/upload.svg" alt="upload CTI file" className="m-auto" width={68} height={68} />
            <p className="text-lg">Drag & Drop to Upload</p>
            <label className="text-sm font-medium text-[#288AFF] cursor-pointer">
              Or Browse
              <input type="file" className="hidden" onChange={handleFileSelect} />
            </label>
          </div>
        </div>
      ) : (
        <div className="border rounded-lg border-dashed border-[rgba(255,255,255,0.4)] h-[244px] p-6 flex flex-col items-center justify-center text-center">
          <p className="text-md font-medium">{file.name} <span className="text-green-400">Uploaded Successfully</span></p>
          <p className="text-sm text-gray-400">{(file.size / 1024).toFixed(2)} KB</p>
          <button className="mt-4 cursor-pointer text-[#9A9A9A]" onClick={handleChooseAgain}>
            Choose Again
          </button>
        </div>
      )}

      <div className="mt-6 flex justify-between">
        <button onClick={onBack} className="px-5 py-2 rounded-xl border cursor-pointer border-[#1272E5] text-[#1272E5]">
          Select Camera
        </button>
        <button
          onClick={onNext}
          disabled={!file}
       className={`px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer disabled:opacity-50`}
        >
          Preview
        </button>
      </div>
    </div>
  );
};

export default CTIUploader;
