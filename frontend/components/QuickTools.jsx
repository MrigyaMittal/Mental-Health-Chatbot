"use client";

import { Wind, Eye, PenLine, Snowflake } from "lucide-react";

export default function QuickTools({ onOpen }) {
  const tools = [
    {
      id: "breathing",
      title: "30s Breathing",
      subtitle: "Calm your body fast",
      icon: <Wind size={18} />,
    },
    {
      id: "grounding",
      title: "5-4-3-2-1",
      subtitle: "Come back to now",
      icon: <Eye size={18} />,
    },
    {
      id: "journal",
      title: "Journaling",
      subtitle: "Clear thoughts gently",
      icon: <PenLine size={18} />,
    },
    {
      id: "sensory",
      title: "Sensory Reset",
      subtitle: "Reset with senses",
      icon: <Snowflake size={18} />,
    },
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {tools.map((t) => (
        <button
          key={t.id}
          onClick={() => onOpen?.(t.id)}
          className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition"
        >
          <span className="text-slate-600">{t.icon}</span>
          <div className="text-left leading-tight">
            <div className="text-sm font-semibold">{t.title}</div>
            <div className="text-[11px] text-slate-500">{t.subtitle}</div>
          </div>
        </button>
      ))}
    </div>
  );
}
