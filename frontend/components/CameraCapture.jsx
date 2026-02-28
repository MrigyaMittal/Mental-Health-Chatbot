"use client";

import { useEffect, useRef, useState } from "react";
import { Video, VideoOff } from "lucide-react";
import * as faceapi from "face-api.js";

const MAP = {
  happy: "joy",
  sad: "sadness",
  angry: "anger",
  surprised: "surprise",
  fearful: "fear",
  disgusted: "disgust",
  neutral: "neutral",
};

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function ema(prev, next, a = 0.25) {
  return prev == null ? next : prev * (1 - a) + next * a;
}

export default function CameraCapture({ onEmotion }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const streamRef = useRef(null);
  const runningRef = useRef(false);

  const lastDetectRef = useRef(0);
  const missRef = useRef(0);

  const stableRef = useRef({
    box: null,
    emotion: "neutral",
    conf: 0.5,
  });

  const winERef = useRef([]);
  const winCRef = useRef([]);

  const [enabled, setEnabled] = useState(false);
  const [emotion, setEmotion] = useState("Not Used");
  const [conf, setConf] = useState(0);

  useEffect(() => {
    let resizeCanvasToVideo = null;

    const loadModels = async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
      await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
      await faceapi.nets.faceExpressionNet.loadFromUri("/models");
    };

    const drawOverlay = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");

      // clear overlay only (video stays stable)
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const show = missRef.current < 10 && stableRef.current.box;
      if (!show) return;

      const { box, emotion, conf } = stableRef.current;

      // face box
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // label background
      const labelW = 220;
      const labelH = 26;
      const lx = box.x;
      const ly = Math.max(0, box.y - 30);

      ctx.fillStyle = "rgba(59, 130, 246, 0.9)";
      ctx.fillRect(lx, ly, labelW, labelH);

      // label text
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 14px system-ui";
      ctx.fillText(`${emotion.toUpperCase()} (${conf.toFixed(2)})`, lx + 8, ly + 18);
    };

    const start = async () => {
      try {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        await loadModels();

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });

        streamRef.current = stream;
        video.srcObject = stream;

        await new Promise((resolve) => {
          video.onloadedmetadata = () => resolve(true);
        });

        await video.play();

        // ✅ MUST sync canvas resolution to video displayed size
        resizeCanvasToVideo = () => {
          const c = canvasRef.current;
          const v = videoRef.current;
          if (!c || !v) return;

          const rect = v.getBoundingClientRect();
          c.width = Math.floor(rect.width);
          c.height = Math.floor(rect.height);
        };

        resizeCanvasToVideo();
        window.addEventListener("resize", resizeCanvasToVideo);

        runningRef.current = true;

        const detectLoop = async (t) => {
          if (!runningRef.current) return;

          const DETECT_MS = 220; // stable + less CPU
          if (t - lastDetectRef.current > DETECT_MS) {
            lastDetectRef.current = t;

            const det = await faceapi
              .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224 }))
              .withFaceLandmarks()
              .withFaceExpressions();

            if (!det) {
              missRef.current += 1;
              drawOverlay();
              requestAnimationFrame(detectLoop);
              return;
            }

            missRef.current = 0;

            // ✅ resize detection results to match canvas/video display size
            const canvasEl = canvasRef.current;
            const dims = faceapi.matchDimensions(canvasEl, video, true);
            const resized = faceapi.resizeResults(det, dims);

            // best expression
            const ex = resized.expressions;
            let bestKey = "neutral";
            let bestVal = 0;

            for (const k in ex) {
              if (ex[k] > bestVal) {
                bestVal = ex[k];
                bestKey = k;
              }
            }

            const mappedEmotion = MAP[bestKey] || "neutral";
            const mappedConf = clamp01(bestVal);

            // rolling smoothing
            winERef.current.push(mappedEmotion);
            winCRef.current.push(mappedConf);
            if (winERef.current.length > 10) winERef.current.shift();
            if (winCRef.current.length > 10) winCRef.current.shift();

            // stable emotion = majority vote
            const counts = {};
            winERef.current.forEach((e) => (counts[e] = (counts[e] || 0) + 1));

            let stableEmotion = mappedEmotion;
            let mx = 0;
            for (const k in counts) {
              if (counts[k] > mx) {
                mx = counts[k];
                stableEmotion = k;
              }
            }

            const avgConf =
              winCRef.current.reduce((a, b) => a + b, 0) / winCRef.current.length;

            // ✅ smooth face box motion
            const b = resized.detection.box;
            const prev = stableRef.current.box;

            const smoothBox = prev
              ? {
                  x: ema(prev.x, b.x),
                  y: ema(prev.y, b.y),
                  width: ema(prev.width, b.width),
                  height: ema(prev.height, b.height),
                }
              : { x: b.x, y: b.y, width: b.width, height: b.height };

            stableRef.current = {
              box: smoothBox,
              emotion: stableEmotion,
              conf: avgConf,
            };

            setEmotion(stableEmotion);
            setConf(Number(avgConf.toFixed(2)));
            onEmotion?.(stableEmotion, Number(avgConf.toFixed(2)));

            drawOverlay();
          }

          requestAnimationFrame(detectLoop);
        };

        requestAnimationFrame(detectLoop);
      } catch (err) {
        console.log(err);
        setEmotion("Camera Error");
        setConf(0);
        onEmotion?.("Not Used", 0);
      }
    };

    const stop = () => {
      runningRef.current = false;

      missRef.current = 0;
      winERef.current = [];
      winCRef.current = [];
      stableRef.current = { box: null, emotion: "neutral", conf: 0.5 };

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      setEmotion("Not Used");
      setConf(0);
      onEmotion?.("Not Used", 0);

      // clear overlay
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      // ✅ remove resize listener
      if (resizeCanvasToVideo) {
        window.removeEventListener("resize", resizeCanvasToVideo);
      }
    };

    if (enabled) start();
    else stop();

    return () => stop();
  }, [enabled, onEmotion]);

  return (
    <div className="space-y-3">
      <button
        onClick={() => setEnabled(!enabled)}
        className={`w-full flex items-center justify-between px-4 py-3 rounded-xl font-medium transition-all ${
          enabled
            ? "bg-blue-500 hover:bg-blue-600 text-white shadow-lg shadow-blue-500/30"
            : "bg-slate-700 hover:bg-slate-600 text-slate-200"
        }`}
      >
        <span className="flex items-center gap-2">
          {enabled ? <Video size={18} /> : <VideoOff size={18} />}
          Emotion Detection
        </span>
        <span className="text-xs opacity-75">{enabled ? "ON" : "OFF"}</span>
      </button>

      {enabled && (
        <div className="space-y-3 animate-in fade-in slide-in-from-top-4 duration-300">
          <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-black shadow-xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
            
            <div className="absolute top-3 left-3 right-3 flex items-center justify-between">
              <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                <span className="text-xs text-white font-medium">LIVE</span>
              </div>
            </div>
          </div>

          <div className="bg-slate-700/50 backdrop-blur-sm rounded-xl p-3 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-300">Detected Emotion</span>
              <span className="font-semibold text-white capitalize">{emotion}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-300">Confidence</span>
              <span className="font-semibold text-white">{(conf * 100).toFixed(0)}%</span>
            </div>
            
            <div className="w-full bg-slate-600 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
                style={{ width: `${conf * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}