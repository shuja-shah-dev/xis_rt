
export default function RoundedButton({ label, active, onClick }) {
  return (
    <div
      onClick={onClick}
      className={`w-[100px] h-[40px] flex items-center justify-center text-[14px] rounded-[118px] cursor-pointer transition
        ${
          active
            ? "bg-[#1272E5] text-white font-semibold"
            : "bg-[rgba(255,255,255,0.05)] text-[#9A9A9A] font-medium"
        }`}
    >
      {label}
    </div>
  );
}
