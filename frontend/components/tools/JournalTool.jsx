"use client";

import { useState } from "react";

export default function JournalTool({ onDone }) {
  const [text, setText] = useState("");

  return (
    <div className="space-y-4">
      <p className="text-slate-600">
        No rules here. Just write freely.
      </p>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-sm text-slate-700">
        ✍️ Prompt: <b>“What’s the heaviest thought in my mind right now?”</b>
        <br />
        Then: <b>“What do I need most in this moment?”</b>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        className="w-full border border-slate-300 rounded-xl px-3 py-2 outline-none focus:ring-2 focus:ring-blue-200"
        placeholder="Write here..."
      />

      <button
        onClick={() =>
          onDone?.(
            "I did a journaling prompt. I feel like I understand my thoughts a bit better now."
          )
        }
        className="w-full bg-slate-900 text-white rounded-xl py-3 font-semibold hover:bg-black transition"
      >
        Done ✅ Add to chat
      </button>
    </div>
  );
}
