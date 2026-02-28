"use client";

import { useState } from "react";

export default function GroundingTool({ onDone }) {
  const [answers, setAnswers] = useState({
    five: "",
    four: "",
    three: "",
    two: "",
    one: "",
  });

  const update = (k, v) => setAnswers((p) => ({ ...p, [k]: v }));

  return (
    <div className="space-y-4">
      <p className="text-slate-600">
        This helps when you feel anxious, spaced out, or overwhelmed.
        Fill whatever you can — no pressure.
      </p>

      <div className="space-y-3">
        <Field
          label="👀 5 things you can SEE"
          value={answers.five}
          onChange={(v) => update("five", v)}
        />
        <Field
          label="✋ 4 things you can FEEL"
          value={answers.four}
          onChange={(v) => update("four", v)}
        />
        <Field
          label="👂 3 things you can HEAR"
          value={answers.three}
          onChange={(v) => update("three", v)}
        />
        <Field
          label="👃 2 things you can SMELL"
          value={answers.two}
          onChange={(v) => update("two", v)}
        />
        <Field
          label="👅 1 thing you can TASTE"
          value={answers.one}
          onChange={(v) => update("one", v)}
        />
      </div>

      <button
        onClick={() =>
          onDone?.(
            "I tried the 5-4-3-2-1 grounding exercise. I feel more present now."
          )
        }
        className="w-full bg-slate-900 text-white rounded-xl py-3 font-semibold hover:bg-black transition"
      >
        Done ✅ Add to chat
      </button>
    </div>
  );
}

function Field({ label, value, onChange }) {
  return (
    <div>
      <div className="text-sm font-semibold text-slate-800 mb-1">{label}</div>
      <input
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        className="w-full border border-slate-300 rounded-xl px-3 py-2 outline-none focus:ring-2 focus:ring-blue-200"
        placeholder="Type here..."
      />
    </div>
  );
}
