import RoundedButton from "./RoundedButton";

export default function Switcher({ activeScreen, onChange }) {
  const tabs = ["Camera", "Model", "Inference"];

  return (
    <div className="text-[#9A9A9A] w-[378px] mt-6 mx-auto flex items-center justify-center gap-3 rounded-[112px] border border-[rgba(255,255,255,0.16)] bg-[rgba(255,255,255,0.05)] backdrop-blur-[28.55px] py-3 px-4">
      {tabs.map((tab) => (
        <RoundedButton
          key={tab}
          label={tab}
          active={activeScreen === tab}
          onClick={() => onChange(tab)}
        />
      ))}
    </div>
  );
}
