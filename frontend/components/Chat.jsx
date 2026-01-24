"use client";

import { useState, useCallback } from "react";
import CameraCapture from "./CameraCapture";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  // ✅ Live emotion from camera
  const [liveEmotion, setLiveEmotion] = useState("Not Used");
  const [liveConf, setLiveConf] = useState(0);

  // ✅ Debug toggle like Streamlit
  const [showDebug, setShowDebug] = useState(true);

  // ✅ FIX: stable callback (prevents CameraCapture flicker)
  const handleEmotion = useCallback((emo, conf) => {
    setLiveEmotion(emo || "Not Used");
    setLiveConf(conf || 0);
  }, []);

  const sendMessage = async () => {
    if (!text.trim() || loading) return;

    const userText = text;
    setText("");
    setLoading(true);

    // ✅ Show user message instantly
    setMessages((prev) => [...prev, { role: "user", text: userText }]);

    // ✅ Build history pairs: [{user, bot}, ...]
    const pairs = [];
    for (let i = 0; i < messages.length - 1; i++) {
      if (messages[i].role === "user" && messages[i + 1].role === "assistant") {
        pairs.push({ user: messages[i].text, bot: messages[i + 1].text });
      }
    }

    const formData = new FormData();
    formData.append("message", userText);
    formData.append("history", JSON.stringify(pairs.slice(-5)));

    // ✅ Send live camera emotion (NO IMAGE)
    formData.append("live_emotion", liveEmotion);
    formData.append("live_conf", String(liveConf));

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Bad response from backend");

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.reply,
          emotion: data.emotion,
          crisis: data.crisis,
          score: data.score,
          text_emotion: data.text_emotion,
          camera_emotion: data.camera_emotion,
          camera_conf: data.camera_conf,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "⚠️ Backend not responding. Please ensure FastAPI is running on port 8000.",
          emotion: "neutral",
          crisis: false,
          score: 0,
          text_emotion: "",
          camera_emotion: "",
          camera_conf: 0,
        },
      ]);
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen w-full bg-slate-50">
      <div className="max-w-6xl mx-auto px-4 py-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* ✅ Sidebar */}
        <div className="md:col-span-1 bg-slate-900 text-white rounded-2xl p-4">
          <h2 className="text-lg font-bold">🧠 Mental Health Support</h2>
          <p className="text-sm text-slate-200 mt-1">Your safe space to talk</p>

          <div className="mt-6">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showDebug}
                onChange={(e) => setShowDebug(e.target.checked)}
              />
              Show emotion & crisis info
            </label>
          </div>

          {/* ✅ Live Camera Emotion */}
          {/* ✅ Live Camera Emotion */}
<div className="mt-6">
  <div className="bg-slate-800 rounded-2xl p-3 border border-slate-700">
    <div className="flex items-start justify-between gap-2">
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-white">
          🎥 Live Emotion Detection
        </h3>
        <p className="text-xs text-slate-300 mt-1 leading-snug break-words">
          Facial mood detection (continuous)
        </p>
      </div>
    </div>

    <div className="mt-3 w-full overflow-hidden rounded-xl border border-slate-700">
      <CameraCapture onEmotion={handleEmotion} />
    </div>

    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
      <span className="text-slate-300">Emotion:</span>
      <span className="px-2 py-1 rounded-full bg-slate-900 text-white font-semibold">
        {liveEmotion}
      </span>

      <span className="text-slate-300">Conf:</span>
      <span className="px-2 py-1 rounded-full bg-slate-900 text-white font-semibold">
        {Number(liveConf).toFixed(2)}
      </span>
    </div>
  </div>
</div>


          {/* ✅ Crisis Support */}
          <div className="mt-6 text-sm">
            <h3 className="font-semibold">🆘 Crisis Support (India)</h3>
            <p className="mt-2">AASRA: 91-9820466726</p>
            <p>iCall: 022-25521111</p>
            <p>Vandrevala: 1860-2662-345</p>
            <p className="mt-2 font-semibold">Emergency: 100</p>
          </div>
        </div>

        {/* ✅ Main Chat */}
        <div className="md:col-span-3 bg-white rounded-2xl shadow p-5 flex flex-col h-[85vh]">
          <div className="text-center mb-4">
            <h1 className="text-2xl font-bold text-blue-700">
              🧠 Mental Health Support
            </h1>
            <p className="text-sm text-slate-500">• A safe space to talk •</p>
          </div>

          {/* ✅ Messages */}
          <div className="flex-1 overflow-y-auto pr-2">
            {messages.length === 0 ? (
              <div className="text-center mt-16 text-slate-400">
                <div className="text-4xl">💭</div>
                <h2 className="text-xl font-semibold text-slate-600 mt-3">
                  Welcome to your safe space
                </h2>
                <p className="mt-2">
                  Share what’s on your mind. I’m here to listen.
                </p>
              </div>
            ) : (
              messages.map((m, idx) => (
                <div key={idx} className="my-3">
                  {m.role === "user" ? (
                    <div className="flex justify-end">
                      <div className="bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-[75%]">
                        {m.text}
                      </div>
                    </div>
                  ) : (
                    <div className="flex justify-start">
                      <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 max-w-[75%] shadow-sm">
                        <div>{m.text}</div>

                        {showDebug && (
                          <div className="mt-3 text-xs text-slate-700 bg-red-50 border-l-4 border-red-400 p-2 rounded">
                            😊 <b>Emotion:</b> {m.emotion} |{" "}
                            {m.crisis ? "🚨 Crisis: Yes" : "✅ Crisis: No"} (Score:{" "}
                            {Number(m.score).toFixed(2)})
                            <br />
                            📝 <b>Text:</b> {m.text_emotion} | 🎥 <b>Cam:</b>{" "}
                            {liveEmotion} ({Number(liveConf).toFixed(2)})
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}

            {/* ✅ Loading bubble */}
            {loading && (
              <div className="flex justify-start my-3">
                <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 max-w-[75%] shadow-sm">
                  💭 Listening...
                </div>
              </div>
            )}
          </div>

          {/* ✅ Input */}
          <div className="mt-4 flex gap-2">
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="💭 Type how you're feeling..."
              className="flex-1 border border-slate-300 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-200"
              onKeyDown={(e) => {
                if (e.key === "Enter") sendMessage();
              }}
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className="bg-blue-600 text-white px-5 rounded-xl font-semibold disabled:opacity-60"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
