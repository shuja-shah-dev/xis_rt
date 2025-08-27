import React, { useState, useRef, useEffect } from "react";
import RoundedButton from "../RoundedButton";

const CameraSelector = ({ onNext }) => {
  const [selected, setSelected] = useState(null);
  const [selectedCameraName, setSelectedCameraName] = useState(""); // New state for camera name
  const [customCameras, setCustomCameras] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newCameraName, setNewCameraName] = useState("");
  const modalRef = useRef(null);

  const baseCameras = [{ id: "video", name: "Video As Webcam" }];

  // Close modal when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (modalRef.current && !modalRef.current.contains(event.target)) {
        setIsModalOpen(false);
      }
    };

    if (isModalOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isModalOpen]);

  const handleSelect = (id, name) => {
    setSelected(id);
    setSelectedCameraName(name); // Store the camera name
  };

  const handleDisconnect = () => {
    setSelected(null);
    setSelectedCameraName(""); // Clear the camera name
  };

  const handleAddCamera = () => {
    if (newCameraName.trim()) {
      const newCamera = {
        id: `custom-${Date.now()}`,
        name: newCameraName,
      };
      setCustomCameras((prev) => [...prev, newCamera]);
      setNewCameraName("");
      setIsModalOpen(false);
    }
  };

  const handleNext = () => {
    onNext(selected, selectedCameraName); // Pass both ID and name
  };

  return (
    <div className="relative flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[106px] py-5 px-8">
      <p className="text-2xl">Choose Your Camera</p>

      <div className="flex gap-10 flex-wrap">
        {/* Base Camera (fixed webcam) */}
        {baseCameras.map((camera) => {
          const isSelected = selected === camera.id;
          const status = isSelected ? "Connected" : "Not Connected";

          return (
            <div
              key={camera.id}
              className={`group w-[335px] h-[160px] rounded-[12px] p-5 backdrop-blur-[110px] transition 
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
                  label={isSelected ? "Selected" : "Select"}
                  active={isSelected}
                  onClick={() => handleSelect(camera.id, camera.name)}
                />
              </div>
            </div>
          );
        })}

        {/* Custom Cameras (insert before + button) */}
        {customCameras.map((camera) => {
          const isSelected = selected === camera.id;
          const status = isSelected ? "Connected" : "Not Connected";

          return (
            <div
              key={camera.id}
              className={`group w-[335px] h-[160px] rounded-[12px] p-5 backdrop-blur-[110px] transition 
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
                  label={isSelected ? "Selected" : "Select"}
                  active={isSelected}
                  onClick={() => handleSelect(camera.id, camera.name)}
                />
              </div>
            </div>
          );
        })}

        {/* + Button (always last) */}
        <div
          onClick={() => setIsModalOpen(true)}
          className="cursor-pointer w-[335px] h-[160px] flex items-center justify-center border-2 rounded-[12px] border-[#0E2332] bg-[rgba(0,0,0,0.04)]  hover:border-[#1272E5]"
        >
          <span className="text-4xl text-[#9A9A9A]">+</span>
        </div>
      </div>

      {/* Modal */}
      {isModalOpen && (
        <div className="absolute top-4 left-[36%] bg-black/50 flex items-center justify-center z-50">
          <div 
            ref={modalRef}
            className="bg-black p-6 rounded-xl w-[400px] flex flex-col gap-4 border-2 border-[#0E2332]"
          >
            <h2 className="text-xl ">Add Camera</h2>
            <input
              type="text"
              placeholder="Camera name"
              value={newCameraName}
              onChange={(e) => setNewCameraName(e.target.value)}
              className="border-2 outline-none border-[#0E2332] rounded-md px-3 py-2"
            />
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setIsModalOpen(false)}
                className="px-4 py-2 rounded bg-gray-400 cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleAddCamera}
                className="px-4 py-2 rounded bg-[#1272E5] text-white cursor-pointer"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="mt-6 flex justify-end">
        <button
          disabled={!selected}
          onClick={handleNext}
          className={`px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer disabled:opacity-50`}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default CameraSelector;