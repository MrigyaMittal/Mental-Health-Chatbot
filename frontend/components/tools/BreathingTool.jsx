"use client";

import { useEffect, useState } from "react";

export default function BreathingTool({ onDone }) {
  const TOTAL = 30;
  const [sec, setSec] = useState(TOTAL);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (!running) return;
    if (sec <= 0) return;

    const t = setTimeout(() => setSec((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [running, sec]);

  const phase = () => {
    // simple loop: inhale 4, hold 2, exhale 4 (repeat)
    const elapsed = TOTAL - sec;
    const cycle = elapsed % 10;

    if (cycle < 4) return "Inhale…";
    if (cycle < 6) return "Hold…";
    return "Exhale…";
  };

  const progress = ((TOTAL - sec) / TOTAL) * 100;

  return (
    <div className="space-y-4">
      <p className="text-slate-600">
        Let’s do a quick <b>30-second breathing reset</b>.
        Follow the rhythm below.
      </p>

      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4">
        <div className="text-2xl font-bold text-slate-800">{phase()}</div>
        <div className="text-sm text-slate-500 mt-1">
          Time left: <b>{sec}s</b>
        </div>

        <div className="w-full h-3 bg-slate-200 rounded-full mt-3 overflow-hidden">
          <div
            className="h-3 bg-blue-500 rounded-full transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="flex gap-2">
        {!running ? (
          <button
            onClick={() => setRunning(true)}
            className="flex-1 bg-blue-600 text-white rounded-xl py-3 font-semibold hover:bg-blue-700 transition"
          >
            Start
          </button>
        ) : (
          <button
            onClick={() => setRunning(false)}
            className="flex-1 bg-slate-200 text-slate-800 rounded-xl py-3 font-semibold hover:bg-slate-300 transition"
          >
            Pause
          </button>
        )}

        <button
          onClick={() => {
            setRunning(false);
            setSec(TOTAL);
          }}
          className="px-4 bg-white border border-slate-200 rounded-xl font-semibold hover:bg-slate-50 transition"
        >
          Reset
        </button>
      </div>

      {sec <= 0 && (
        <div className="p-3 bg-green-50 border border-green-200 rounded-xl text-green-800 text-sm">
          ✅ Great job. Your nervous system just got a small reset.
        </div>
      )}

      <button
        onClick={() =>
          onDone?.("I did the 30s breathing exercise. I feel a bit calmer now.")
        }
        className="w-full bg-slate-900 text-white rounded-xl py-3 font-semibold hover:bg-black transition"
      >
        Done ✅ Add to chat
      </button>
    </div>
  );
}
