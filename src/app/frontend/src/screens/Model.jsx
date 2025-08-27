import React, { useState } from "react";
import RoundedButton from "../components/RoundedButton";

const Model = ({ setActiveScreen }) => {
  const [selected, setSelected] = useState(null);

  const categories = [
    {
      name: "Generic",
      models: [{ id: "general", name: "General Model" }],
    },
    {
      name: "Baked Baguette",
      models: [{ id: "Baguette-twin", name: "Digital Twin Model" }],
    },
    {
      name: "Raw Dough",
      models: [{ id: "raw-twin", name: "Digital Twin Model" }],
    },
  ];

  const handleSelect = (id) => setSelected(id);
  const handleDisconnect = () => setSelected(null);

  return (
    <div>
      <p className="text-3xl font-semibold">Model Configuration</p>
      <p className="text-sm text-[#73768D]">
        Select desired model from the following categories.
      </p>

      <div className="flex gap-6 mt-6">
        {categories.slice(0, 2).map((category) => (
          <div
            key={category.name}
            className="relative flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-5 px-8 w-1/2"
          >
            <p className="text-2xl">{category.name}</p>

            <div className="flex gap-10 flex-wrap">
              {category.models.map((model) => {
                const isSelected = selected === model.id;

                return (
                  <div
                    key={model.id}
                    className={`group w-[335px] h-[160px] border-[0.5px] rounded-[12px] p-5 backdrop-blur-[110.57px] transition 
                      ${
                        isSelected
                          ? "border-[#1272E5]"
                          : "border-[rgba(255,255,255,0.16)] bg-[rgba(0,0,0,0.03)] hover:border-[#1272E5]"
                      }`}
                  >
                    <div className="flex justify-between items-center">
                      <p className="text-lg font-medium">{model.name}</p>

                      <div className="flex items-center gap-2">
                        {isSelected && (
                          <div
                            className="cursor-pointer"
                            onClick={handleDisconnect}
                          >
                            <img
                              src="/disconnector.svg"
                              alt="Disconnector"
                              width={21}
                              height={21}
                            />
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="mt-2 h-[0.5px] bg-[linear-gradient(90deg,rgba(255,255,255,0)_0%,rgba(255,255,255,0.31)_52.4%,rgba(255,255,255,0)_100%)]"></div>

                    <div className="mt-6">
                      <RoundedButton
                        label="Select"
                        active={isSelected}
                        onClick={() => handleSelect(model.id)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Raw Dough Category */}
      <div
        key={categories[2].name}
        className="relative flex flex-col gap-8 border-2 rounded-[12px] border-[#0E2332] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-5 px-8 mt-6 w-full"
      >
        <p className="text-2xl">{categories[2].name}</p>

        <div className="flex gap-10 flex-wrap">
          {categories[2].models.map((model) => {
            const isSelected = selected === model.id;

            return (
              <div
                key={model.id}
                className={`group w-[335px] h-[160px] border-[0.5px] rounded-[12px] p-5 backdrop-blur-[110.57px] transition 
                  ${
                    isSelected
                      ? "border-[#1272E5]"
                      : "border-[rgba(255,255,255,0.16)] bg-[rgba(0,0,0,0.03)] hover:border-[#1272E5]"
                  }`}
              >
                <div className="flex justify-between items-center">
                  <p className="text-lg font-medium">{model.name}</p>

                  <div className="flex items-center gap-2">
                    {isSelected && (
                      <div
                        className="cursor-pointer"
                        onClick={handleDisconnect}
                      >
                        <img
                          src="/disconnector.svg"
                          alt="Disconnector"
                          width={21}
                          height={21}
                        />
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-2 h-[0.5px] bg-[linear-gradient(90deg,rgba(255,255,255,0)_0%,rgba(255,255,255,0.31)_52.4%,rgba(255,255,255,0)_100%)]"></div>

                <div className="mt-6">
                  <RoundedButton
                    label="Select"
                    active={isSelected}
                    onClick={() => handleSelect(model.id)}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Next Button */}
      <div className="mt-6 flex justify-end">
        <button
          disabled={!selected}
          onClick={() => setActiveScreen("Inference")}
          className="px-5 py-2 rounded-xl bg-[#1272E5] text-white text-md cursor-pointer disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default Model;
