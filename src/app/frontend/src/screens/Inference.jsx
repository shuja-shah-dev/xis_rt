import React from "react";
import { Blob, BlobLarge } from "../components/Blob";


const Inference = () => {
  return (
    <div>
      
      <p className="text-3xl font-semibold">Inference</p>
      <p className="text-sm text-[#73768D]">
        Run real-time predictions on feed using the selected trained model
      </p>

      <div className="relative h-[450px] border-2 border-[#0E2332] bg-[rgba(0,0,0,0.11)] backdrop-blur-[106.0999984741211px] rounded-xl p-4 flex items-center justify-center mt-6">
         {/* <Blob className="top-0 left-0" />
          <Blob className="bottom-0 right-0" /> */}
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
      </div>

      <div className="mt-6 flex gap-4 justify-end">
        <button className="px-5 py-2 rounded-xl bg-[#4061A9]">Snapshot</button>
        <button
          className={`px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer`}
        >
          Start Inference
        </button>
      </div>
    </div>
  );
};

export default Inference;
