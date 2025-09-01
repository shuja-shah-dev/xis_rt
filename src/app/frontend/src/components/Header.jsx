import React from "react";

const StatusBadge = ({ label, status }) => {
  const colors = {
    Connected: "bg-[#1AA64B]",
    Loaded: "bg-[#1AA64B]",
    "Not Loaded": "bg-[#FF4C40]",
    Disconnected: "bg-[#FF4C40]",
  };

  return (
    <div className="flex items-center justify-between w-[187px] rounded-[12px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-5">
      <p className="text-[14px] font-semibold mr-2">{label}</p>
      <div className={`text-[10px] font-semibold rounded-4xl py-0.5 px-4 ${colors[status]}`}>
        {status}
      </div>
    </div>
  );
};

const Header = ({ modelStatus, mqttStatus, cameraStatus }) => {
 
  return (
    <div className="flex items-center justify-between rounded-[12px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-4">
      <div>
        <img src="/logo.svg" alt="xisRT Logo" width={99} height={34} />
      </div>
      <div className="flex gap-2">
        <StatusBadge label="Model" status={modelStatus} />
        <StatusBadge label="MQTT" status={mqttStatus} />
        <StatusBadge label="Camera" status={cameraStatus} />
      </div>
    </div>
  );
};

export default Header;
