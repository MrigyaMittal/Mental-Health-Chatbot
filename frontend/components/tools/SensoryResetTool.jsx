"use client";

import { useState } from "react";

export default function SensoryResetTool({ onDone }) {
  const [checked, setChecked] = useState({
    water: false,
    feet: false,
    shoulders: false,
    eyes: false,
  });

  const toggle = (k) => setChecked((p) => ({ ...p, [k]: !p[k] }));

  return (
    <div className="space-y-4">
      <p className="text-slate-600">
        Pick 1–2 small actions. These calm your body quickly.
      </p>

      <div className="space-y-2">
        <CheckItem
          label="🧊 Splash cold water / hold something cold for 10s"
          checked={checked.water}
          onClick={() => toggle("water")}
        />
        <CheckItem
          label="👣 Press your feet into the floor for 10s"
          checked={checked.feet}
          onClick={() => toggle("feet")}
        />
        <CheckItem
          label="🤲 Drop your shoulders + unclench your jaw"
          checked={checked.shoulders}
          onClick={() => toggle("shoulders")}
        />
        <CheckItem
          label="👀 Slowly look left → right (ground your eyes)"
          checked={checked.eyes}
          onClick={() => toggle("eyes")}
        />
      </div>

      <button
        onClick={() =>
          onDone?.("I did a sensory reset. My body feels less tense now.")
        }
        className="w-full bg-slate-900 text-white rounded-xl py-3 font-semibold hover:bg-black transition"
      >
        Done ✅ Add to chat
      </button>
    </div>
  );
}

function CheckItem({ label, checked, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-3 rounded-xl border transition ${
        checked
          ? "bg-green-50 border-green-200 text-green-800"
          : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
      }`}
    >
      <div className="flex items-center gap-2">
        <div
          className={`w-4 h-4 rounded border flex items-center justify-center ${
            checked ? "bg-green-500 border-green-500" : "border-slate-300"
          }`}
        >
          {checked ? <span className="text-white text-xs">✓</span> : null}
        </div>
        <div className="text-sm">{label}</div>
      </div>
    </button>
  );
}
