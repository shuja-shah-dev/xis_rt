import React, { useEffect, useState } from "react";
import RoundedButton from "../components/RoundedButton";

const Model = ({ setActiveScreen, selectedVideo, setModelStatus , setShowMeasurement}) => {
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState(null);


  const categories = [
    { name: "Generic", models: [{ id: "generic", name: "General Model" }] },
    { name: "Baked Baguette", models: [{ id: "baked_baguette", name: "Baked Baguette Model" }] },
    { name: "Raw Dough", models: [{ id: "raw_dough", name: "Digital Twin Model" }] },
    { name: "Donut", models: [{ id: "donut", name: "Donut Model" }] },
  ];

  useEffect(() => {
    const savedSelected = localStorage.getItem("selectedCamera");
    if (savedSelected) {
      const { name } = JSON.parse(savedSelected);
      setSelectedCamera(name);
    }
  }, []);

  const videoToModelMap = {
    raw: "raw_dough",
    baked: "baked_baguette",
    donut: "donut",
  };

  let allowedModelId = null;
  let restrictByVideo = false

  if (selectedCamera === "Video As Webcam") {
 
    restrictByVideo = selectedVideo !== null && selectedVideo !== undefined;
    allowedModelId = restrictByVideo ? videoToModelMap[selectedVideo] ?? null : null;
  } else {

    allowedModelId = "generic";
    restrictByVideo = true; 
  }

  const handleSelect = async (modelId) => {
    if (restrictByVideo && modelId !== allowedModelId) {

      setError("This model is disabled for the selected video.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:5000/api/select_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelId }),
      });
      const data = await response.json();
      if (!response.ok || data.status === "error") {
        throw new Error(data.message || "Model selection failed");
      }
      setSelected(modelId);
      setModelStatus("Loaded");
    } catch (e) {
      setSelected(null);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <p className="text-3xl font-semibold">Select AI Model/Recipe</p>
      <p className="text-sm text-[#73768D]">
        Select desired model from the following categories.
      </p>

      <div className="grid grid-cols-2 gap-6 mt-6">
        {categories.map((category) => (
          <div
            key={category.name}
            className="relative flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-5 px-8"
          >
            <p className="text-2xl">{category.name}</p>

            <div className="grid grid-cols-2 gap-6">
              {category.models.map((model) => {
                const isSelected = selected === model.id;
                const isAllowed =
                  !restrictByVideo || model.id === allowedModelId;

                return (
                  <div
                    key={model.id}
                    className={`relative group h-[160px] w-[470px] border-[0.5px] rounded-[12px] p-5 backdrop-blur-[110.57px] transition
                      ${isSelected ? "border-[#1272E5]" : "border-[rgba(255,255,255,0.16)] bg-[rgba(0,0,0,0.03)] hover:border-[#1272E5]"}
                      ${!isAllowed ? "opacity-50 " : ""}
                    `}
                    aria-disabled={!isAllowed}

                  >
                    {!isAllowed && (
                      <div className="absolute -top-10 left-1/2 -translate-x-1/2 hidden group-hover:block bg-black text-white text-xs rounded-md px-3 py-1 z-20">
                        Disabled for the selected video
                        <div className="absolute left-1/2 -bottom-1.5 -translate-x-1/2 w-2 h-2 bg-black rotate-45"></div>
                      </div>
                    )}

                    <div className="flex justify-between items-center">
                      <p className="text-lg font-medium">{model.name}</p>
                      {isSelected && (
                        <div className="cursor-pointer" onClick={() => setSelected(null)}>
                          <img src="/disconnector.svg" alt="Disconnector" width={21} height={21} />
                        </div>
                      )}
                    </div>

                    <div className="mt-6 ">

                      <RoundedButton
                        label={loading && isSelected ? "Selecting..." : "Select"}
                        active={isSelected}
                        onClick={() => {
                          if (!isAllowed || loading) return;
                          handleSelect(model.id);
                        }}
                        disabled={loading || !isAllowed}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}

      <div className="mt-6 flex justify-end">
        <button
          disabled={!selected}
          onClick={() => {setShowMeasurement(true); setActiveScreen("Inference")}}
          className="px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
};


export default Model;
