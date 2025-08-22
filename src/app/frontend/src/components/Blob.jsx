import React from "react";

const Blob = ({ className = "" }) => {
  return (
    <div
      className={`absolute w-[499px] h-[499px] rounded-full 
        bg-[linear-gradient(138deg,rgba(18,114,229,0.35)_-3.48%,rgba(10,63,127,0)_98.53%)]
        blur-[105px] ${className}`}
    />
  );
};

const BlobLarge = ({ className = "" }) => {
  return (
    <div
      className={`absolute w-full h-[60vh] rounded-full 
        bg-[rgba(19,43,97,0.46)] blur-[250px] -z-10 ${className}`}
    />
  );
};

export { Blob, BlobLarge };
