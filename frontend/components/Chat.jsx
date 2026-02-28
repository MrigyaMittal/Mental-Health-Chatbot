"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Send, Settings, AlertCircle, Phone } from "lucide-react";

import CameraCapture from "./CameraCapture";
import QuickTools from "./QuickTools";
import ToolModal from "./ToolModal";

import BreathingTool from "./tools/BreathingTool";
import GroundingTool from "./tools/GroundingTool";
import JournalTool from "./tools/JournalTool";
import SensoryResetTool from "./tools/SensoryResetTool";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  // ✅ Live emotion from camera
  const [liveEmotion, setLiveEmotion] = useState("Not Used");
  const [liveConf, setLiveConf] = useState(0);

  // ✅ Debug toggle
  const [showDebug, setShowDebug] = useState(true);

  // ✅ Sidebar toggle
  const [showSidebar, setShowSidebar] = useState(true);

  // ✅ Quick Tools modal state
  const [toolOpen, setToolOpen] = useState(false);
  const [activeTool, setActiveTool] = useState(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // ✅ stable callback (prevents CameraCapture flicker)
  const handleEmotion = useCallback((emo, conf) => {
    setLiveEmotion(emo || "Not Used");
    setLiveConf(conf || 0);
  }, []);

  // ✅ open/close tool modal
  const openTool = (toolId) => {
    setActiveTool(toolId);
    setToolOpen(true);
  };

  const closeTool = () => {
    setToolOpen(false);
    setActiveTool(null);
  };

  // ✅ when a tool is done, put text into chat input (user can edit)
  const toolDone = (msg) => {
    closeTool();
    setText(msg);
  };

  const sendMessage = async () => {
    if (!text.trim() || loading) return;

    const userText = text;
    setText("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: userText }]);

    // ✅ Build history pairs
    const pairs = [];
    for (let i = 0; i < messages.length - 1; i++) {
      if (messages[i].role === "user" && messages[i + 1].role === "assistant") {
        pairs.push({ user: messages[i].text, bot: messages[i + 1].text });
      }
    }

    const formData = new FormData();
    formData.append("message", userText);
    formData.append("history", JSON.stringify(pairs.slice(-5)));
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

  const EmotionBadge = ({ emotion }) => {
    const colors = {
      joy: "bg-yellow-100 text-yellow-800 border-yellow-200",
      sadness: "bg-blue-100 text-blue-800 border-blue-200",
      anger: "bg-red-100 text-red-800 border-red-200",
      fear: "bg-purple-100 text-purple-800 border-purple-200",
      surprise: "bg-pink-100 text-pink-800 border-pink-200",
      disgust: "bg-green-100 text-green-800 border-green-200",
      neutral: "bg-slate-100 text-slate-800 border-slate-200",
    };

    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
          colors[emotion] || colors.neutral
        }`}
      >
        {emotion}
      </span>
    );
  };

  const modalTitle =
    activeTool === "breathing"
      ? "🌬️ 30s Breathing"
      : activeTool === "grounding"
      ? "👀 5-4-3-2-1 Grounding"
      : activeTool === "journal"
      ? "✍️ Journaling Prompt"
      : activeTool === "sensory"
      ? "🧊 Sensory Reset"
      : "Tool";

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* ✅ Sidebar */}
      <div
        className={`${
          showSidebar ? "w-80" : "w-0"
        } transition-all duration-300 overflow-hidden`}
      >
        <div className="h-full bg-gradient-to-b from-slate-800 to-slate-900 text-white p-6 flex flex-col shadow-2xl overflow-y-auto">
          {/* ✅ Title */}
          <div className="space-y-2 mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              MindCare AI
            </h2>
            <p className="text-slate-400 text-sm">Your compassionate companion</p>
          </div>

          {/* ✅ Camera Box */}
          <div className="mb-4">
            <div className="rounded-xl bg-slate-700/20 border border-slate-600/30 p-3">
              <CameraCapture onEmotion={handleEmotion} />

              {/* ✅ Live Emotion pill */}
              <div className="mt-3 flex items-center justify-between gap-2">
                <div className="text-xs text-slate-400">Live Emotion</div>

                <div className="max-w-[170px] flex items-center gap-2 px-2 py-1 rounded-full bg-slate-900/60 border border-slate-700 overflow-hidden">
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse shrink-0" />
                  <span className="text-xs text-white font-semibold truncate capitalize">
                    {liveEmotion}
                  </span>
                  <span className="text-[10px] text-slate-300 shrink-0">
                    {Number(liveConf).toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* ✅ Settings */}
          <div className="mb-6 p-4 bg-slate-700/30 rounded-xl backdrop-blur-sm border border-slate-600/30">
            <div className="flex items-center gap-2 mb-3">
              <Settings size={16} className="text-slate-400" />
              <span className="text-sm font-medium">Settings</span>
            </div>

            <label className="flex items-center gap-3 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={showDebug}
                onChange={(e) => setShowDebug(e.target.checked)}
                className="w-4 h-4 rounded accent-blue-500"
              />
              <span className="text-slate-300">Show emotion & crisis info</span>
            </label>
          </div>

          {/* ✅ Crisis Support */}
          <div className="mt-auto space-y-4">
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
              <div className="flex items-center gap-2 mb-3">
                <AlertCircle size={18} className="text-red-400" />
                <h3 className="font-semibold text-red-200">Crisis Support (India)</h3>
              </div>

              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex items-center gap-2">
                  <Phone size={14} className="shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium text-white truncate">AASRA</div>
                    <div className="text-xs text-slate-300 truncate">91-9820466726</div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Phone size={14} className="shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium text-white truncate">iCall</div>
                    <div className="text-xs text-slate-300 truncate">022-25521111</div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Phone size={14} className="shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium text-white truncate">Vandrevala</div>
                    <div className="text-xs text-slate-300 truncate">1860-2662-345</div>
                  </div>
                </div>

                <div className="mt-2 pt-2 border-t border-slate-700">
                  <div className="font-semibold text-white">Emergency: 100</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ✅ Main Chat */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-slate-200 px-6 py-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowSidebar(!showSidebar)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>

              <div>
                <h1 className="text-xl font-bold text-slate-800">
                  🧠 Mental Health Support
                </h1>
                <p className="text-sm text-slate-500">A safe space to talk</p>
              </div>
            </div>

            {liveEmotion !== "Not Used" && (
              <div className="hidden sm:flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-full border border-blue-200">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-blue-700 capitalize truncate max-w-[140px]">
                  {liveEmotion}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <svg
                    className="w-12 h-12 text-blue-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                    />
                  </svg>
                </div>

                <h2 className="text-2xl font-bold text-slate-700 mb-2">
                  Welcome to your safe space
                </h2>
                <p className="text-slate-500 mb-6">
                  Share what&apos;s on your mind. I&apos;m here to listen.
                </p>

                <div className="flex flex-wrap gap-2 justify-center">
                  <button
                    onClick={() => setText("I'm feeling stressed today")}
                    className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm hover:bg-slate-50 transition-colors"
                  >
                    I&apos;m feeling stressed
                  </button>
                  <button
                    onClick={() => setText("I need someone to talk to")}
                    className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm hover:bg-slate-50 transition-colors"
                  >
                    Need to talk
                  </button>
                  <button
                    onClick={() => setText("How can you help me?")}
                    className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm hover:bg-slate-50 transition-colors"
                  >
                    How can you help?
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    m.role === "user" ? "justify-end" : "justify-start"
                  } animate-in fade-in slide-in-from-bottom-4 duration-300`}
                >
                  {m.role === "user" ? (
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-2xl rounded-tr-sm px-5 py-3 max-w-[75%] shadow-lg shadow-blue-500/30">
                      <p className="leading-relaxed">{m.text}</p>
                    </div>
                  ) : (
                    <div className="max-w-[75%]">
                      <div className="bg-white rounded-2xl rounded-tl-sm px-5 py-3 shadow-md border border-slate-100">
                        <p className="leading-relaxed text-slate-800">{m.text}</p>

                        {showDebug && (
                          <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
                            <div className="flex items-center gap-2 flex-wrap text-xs">
                              <span className="text-slate-600">Detected:</span>
                              <EmotionBadge emotion={m.emotion} />

                              {m.crisis && (
                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800 border border-red-200">
                                  <AlertCircle size={12} className="mr-1" />
                                  Crisis Alert
                                </span>
                              )}

                              <span className="text-slate-500">
                                Crisis Score: {Number(m.score).toFixed(2)}
                              </span>
                            </div>

                            <div className="text-xs text-slate-600 space-y-1">
                              <div>
                                📝 <b>Text:</b> {m.text_emotion}
                              </div>
                              <div>
                                🎥 <b>Cam:</b> {m.camera_emotion} (
                                {Number(m.camera_conf).toFixed(2)})
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="flex justify-start animate-in fade-in slide-in-from-bottom-4 duration-300">
                  <div className="bg-white rounded-2xl rounded-tl-sm px-5 py-3 shadow-md border border-slate-100">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div
                          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0ms" }}
                        />
                        <div
                          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                          style={{ animationDelay: "150ms" }}
                        />
                        <div
                          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                          style={{ animationDelay: "300ms" }}
                        />
                      </div>
                      <span className="text-sm text-slate-500">💭 Listening...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* ✅ Quick Tools row */}
        <div className="px-6 pb-2">
          <div className="max-w-3xl mx-auto">
            <QuickTools onOpen={openTool} />
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-slate-200 px-6 py-4 shadow-lg">
          <div className="max-w-3xl mx-auto">
            <div className="flex gap-3 items-end">
              <div className="flex-1 relative">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="💭 Type how you're feeling..."
                  className="w-full resize-none border border-slate-300 rounded-2xl px-5 py-3 pr-12 outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all max-h-32 bg-slate-50"
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  onInput={(e) => {
                    e.target.style.height = "auto";
                    e.target.style.height = e.target.scrollHeight + "px";
                  }}
                />
              </div>

              <button
                onClick={sendMessage}
                disabled={loading || !text.trim()}
                className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-3.5 rounded-2xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-blue-500/30 transition-all active:scale-95"
              >
                <Send size={20} />
              </button>
            </div>

            <p className="text-xs text-slate-400 mt-2 text-center">
              Press Enter to send • Shift + Enter for new line
            </p>
          </div>
        </div>
      </div>

      {/* ✅ Tool Modal */}
      <ToolModal open={toolOpen} title={modalTitle} onClose={closeTool}>
        {activeTool === "breathing" && <BreathingTool onDone={toolDone} />}
        {activeTool === "grounding" && <GroundingTool onDone={toolDone} />}
        {activeTool === "journal" && <JournalTool onDone={toolDone} />}
        {activeTool === "sensory" && <SensoryResetTool onDone={toolDone} />}
      </ToolModal>
    </div>
  );
}
