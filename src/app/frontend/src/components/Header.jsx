import React from "react";

const Header = () => {
  return (
    <div className="flex items-center justify-between rounded-[12px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-4">
      <div>
        <img src="/logo.svg" alt="xisRT Logo" width={99} height={34}/>
      </div>
      <div className="flex gap-2">
        <div className="flex items-center justify-between w-[187px] rounded-[12px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-5">
            <p className="text-[14px] font-semibold">Model</p>
            <div className="text-[10px] font-semibold rounded-4xl bg-[#FF4C40] py-1 px-5">Not Loaded</div>
        </div>

           <div className="flex items-center justify-between w-[187px] rounded-[12px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-5">
            <p className="text-[14px] font-semibold">MQTT</p>
            <div className="text-[10px] font-semibold rounded-4xl bg-[#FF4C40] py-0.5 px-5">Not Loaded</div>
        </div>

           <div className="flex items-center justify-between w-[187px] rounded-[12px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-5">
            <p className="text-[14px] font-semibold">Camera</p>
            <div className="text-[10px] font-semibold rounded-4xl bg-[#1AA64B] py-0.5 px-5">Connected</div>
        </div>
      </div>
    </div>
  );
};

export default Header;
