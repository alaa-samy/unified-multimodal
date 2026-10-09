"use client";

import { useState, useRef, useEffect } from "react";
import { Square } from "lucide-react";

interface UnifiedSearchInputProps {
  onSearch: (data: { query: string; image: File | null; voiceText: string }) => void;
  loading: boolean;
  searchCancelable: boolean;
  onCancelSearch: () => void;
}

interface TranscriptionResponse {
  ok?: boolean;
  text?: string;
  error?: string | null;
  warning?: string | null;
  looks_like_search?: boolean;
}

export function UnifiedSearchInput({ onSearch, loading, searchCancelable, onCancelSearch }: UnifiedSearchInputProps) {
  const [query, setQuery]               = useState("");
  const [imageFiles, setImageFiles]     = useState<File[]>([]);
  const [recording, setRecording]       = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [voiceText, setVoiceText]       = useState("");
  const [voiceWarning, setVoiceWarning] = useState("");
  const [voiceError, setVoiceError]     = useState("");
  const [hasVoiceAttachment, setHasVoiceAttachment] = useState(false);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef   = useRef<Blob[]>([]);
  const fileInputRef     = useRef<HTMLInputElement>(null);
  const wavFileRef       = useRef<File | null>(null);
  const rawBlobRef       = useRef<Blob | null>(null);

  useEffect(() => () => {
    const recorder = mediaRecorderRef.current;
    if (recorder) {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      if (recorder.state !== "inactive") recorder.stop();
      recorder.stream.getTracks().forEach(track => track.stop());
      mediaRecorderRef.current = null;
    }
  }, []);

  // ── Convert any audio blob → 16kHz mono WAV in browser (no ffmpeg) ──
  async function blobToWav(blob: Blob): Promise<File> {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx    = new AudioContext();
    try {
      const decoded     = await audioCtx.decodeAudioData(arrayBuffer);
      const targetSr    = 16000;
      const offCtx      = new OfflineAudioContext(1, Math.ceil(decoded.duration * targetSr), targetSr);
      const src         = offCtx.createBufferSource();
      src.buffer        = decoded;
      src.connect(offCtx.destination);
      src.start(0);
      const rendered = await offCtx.startRendering();
      const samples  = rendered.getChannelData(0);
      return new File([encodeWav(samples, targetSr)], "recording.wav", { type: "audio/wav" });
    } finally {
      await audioCtx.close();
    }
  }

  function encodeWav(samples: Float32Array, sr: number): ArrayBuffer {
    const buf  = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buf);
    const str  = (off: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
    str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
    str(8, "WAVE"); str(12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);  view.setUint16(34, 16, true);
    str(36, "data"); view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return buf;
  }

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  function clearVoice() {
    setVoiceText("");
    setVoiceWarning("");
    setVoiceError("");
    setHasVoiceAttachment(false);
    setTranscribing(false);
    wavFileRef.current = null;
    rawBlobRef.current = null;
  }

  async function transcribeWav(wav: File) {
    wavFileRef.current = wav;
    setHasVoiceAttachment(true);
    setVoiceError("");
    setTranscribing(true);

    try {
      const form = new FormData();
      form.append("audio", wav);
      const response = await fetch(`${apiUrl}/transcribe`, { method: "POST", body: form });
      const data = await response.json().catch(() => null) as TranscriptionResponse | null;
      if (!response.ok || !data?.ok) {
        const detail = data?.error?.trim();
        throw new Error(detail || `Transcription failed (${response.status || "unknown status"})`);
      }

      setVoiceText(typeof data.text === "string" ? data.text : "");
      setVoiceWarning(typeof data.warning === "string" ? data.warning : "");
      setVoiceError(typeof data.error === "string" ? data.error : "");
      if (data.looks_like_search === false && !data.warning) {
        setVoiceWarning("This may not be a furniture search. You can edit it before searching.");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setVoiceText("");
      setVoiceWarning("");
      setVoiceError(message.toLowerCase().includes("failed to fetch")
        ? `I can't reach the search server at ${apiUrl}. Please check that the backend is running.`
        : message);
    } finally {
      setTranscribing(false);
    }
  }

  async function prepareAndTranscribe(blob: Blob) {
    rawBlobRef.current = blob;
    setHasVoiceAttachment(true);
    setTranscribing(true);
    setVoiceError("");
    try {
      const wav = await blobToWav(blob);
      await transcribeWav(wav);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setVoiceError(message);
      setTranscribing(false);
    }
  }

  // ── Voice Recording ──────────────────────────────────────────
  async function startRecording() {
    clearVoice();
    let stream: MediaStream | null = null;
    try {
      const recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream = recordingStream;
      const mimeType = ["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/mp4"]
        .find(m => MediaRecorder.isTypeSupported(m)) || "";
      mediaRecorderRef.current = new MediaRecorder(recordingStream, mimeType ? { mimeType } : {});
      audioChunksRef.current   = [];

      mediaRecorderRef.current.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mediaRecorderRef.current.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current?.mimeType || "audio/webm" });
        recordingStream.getTracks().forEach(t => t.stop());
        mediaRecorderRef.current = null;
        await prepareAndTranscribe(rawBlob);
      };
      mediaRecorderRef.current.start(100);
      setRecording(true);
    } catch {
      stream?.getTracks().forEach(track => track.stop());
      alert("Could not access microphone. Please allow permissions.");
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  }

  // ── Image ────────────────────────────────────────────────────
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setImageFiles(prev => [...prev, ...files]);
    
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = ev => {
        setImagePreviews(prev => [...prev, ev.target?.result as string]);
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (index: number) => { 
    setImageFiles(prev => prev.filter((_, i) => i !== index)); 
    setImagePreviews(prev => prev.filter((_, i) => i !== index)); 
    if (fileInputRef.current && imageFiles.length === 1) fileInputRef.current.value = ""; 
  };
  // ── Clear everything after submit ────────────────────────────
  function clearAll() {
    setQuery("");
    setImageFiles([]);
    setImagePreviews([]);
    clearVoice();
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // ── Submit ───────────────────────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedVoiceText = voiceText.trim();
    if (!query.trim() && imageFiles.length === 0 && !trimmedVoiceText) return;

    // Capture current values before clearing
    const submitData = { query: query.trim(), image: imageFiles[0] || null, voiceText: trimmedVoiceText };

    // Clear inputs immediately so UI feels responsive
    clearAll();

    // Fire search
    onSearch(submitData);
  };

  const activeInputs = [query.trim() && "text", imageFiles.length > 0 && "image", voiceText.trim() && "voice"].filter(Boolean);

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="space-y-3">

        {/* Attachments */}
        {(imageFiles.length > 0 || hasVoiceAttachment || transcribing) && (
          <div className="flex flex-wrap gap-2 p-3 bg-gray-50 rounded-xl border border-gray-200">
            {imageFiles.map((file, i) => (
              <div key={i} className="flex items-center gap-2 bg-white rounded-lg p-2 border border-gray-200 shadow-sm">
                {imagePreviews[i] && <img src={imagePreviews[i]} alt="Preview" className="w-10 h-10 object-cover rounded" />}
                <div className="flex flex-col min-w-0">
                  <span className="text-xs font-medium truncate max-w-[120px]">{file.name}</span>
                  <span className="text-xs text-gray-400">{Math.round(file.size / 1024)}KB</span>
                </div>
                <button type="button" onClick={() => removeImage(i)} className="text-red-400 hover:text-red-600 font-bold ml-1">×</button>
              </div>
            ))}
            {(hasVoiceAttachment || transcribing) && (
              <div className="min-w-[min(100%,20rem)] flex-1 bg-white rounded-lg p-3 border border-green-200 shadow-sm">
                <div className="flex items-center gap-2">
                  <span className="text-lg" aria-hidden="true">🎤</span>
                  {transcribing ? (
                    <span className="flex items-center gap-2 text-xs text-gray-600">
                      <span className="w-3 h-3 border-2 border-violet-500 border-t-transparent rounded-full animate-spin inline-block" />
                      Transcribing...
                    </span>
                  ) : (
                    <input
                      aria-label="Edit voice transcript"
                      value={voiceText}
                      onChange={event => setVoiceText(event.target.value)}
                      placeholder="Edit what you said..."
                      className="min-w-0 flex-1 border border-gray-200 rounded-md px-2 py-1 text-sm outline-none focus:border-violet-400"
                    />
                  )}
                  <button type="button" onClick={clearVoice} className="text-red-400 hover:text-red-600 font-bold" aria-label="Clear voice transcript">×</button>
                </div>
                {!transcribing && (
                  <div className="mt-2 flex items-center justify-between gap-3">
                    {voiceError ? (
                      <button type="button" onClick={() => wavFileRef.current ? transcribeWav(wavFileRef.current) : rawBlobRef.current && prepareAndTranscribe(rawBlobRef.current)} className="text-xs font-medium text-violet-700 hover:text-violet-900">
                        Try again
                      </button>
                    ) : <span />}
                    <button type="button" onClick={startRecording} disabled={loading || recording} className="text-xs font-medium text-violet-700 hover:text-violet-900 disabled:opacity-40">
                      Record again
                    </button>
                  </div>
                )}
                {voiceWarning && <p className="mt-2 text-xs text-amber-700">{voiceWarning}</p>}
                {voiceError && <p className="mt-2 text-xs text-red-600">{voiceError}</p>}
              </div>
            )}
          </div>
        )}

        {activeInputs.length > 1 && (
          <div className="flex justify-center">
            <span className="inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-medium bg-violet-100 text-violet-700 border border-violet-200">
              ✨ Combined: {activeInputs.join(" + ")}
            </span>
          </div>
        )}

        {/* Input bar */}
        <div className="bg-white rounded-xl border-2 border-gray-200 focus-within:border-violet-400 transition-colors shadow-sm">
          <div className="flex items-center gap-2 px-3 py-3">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder='Type, upload image, or record voice...'
              className="flex-1 border-0 bg-transparent text-sm outline-none placeholder:text-gray-400"
              disabled={loading}
            />

            <input ref={fileInputRef} type="file" multiple accept="image/*" onChange={handleImageUpload} className="hidden" />
            <button type="button" onClick={() => fileInputRef.current?.click()} disabled={loading}
                    className={`p-1.5 rounded-lg text-lg transition-colors ${imageFiles.length > 0 ? "bg-blue-100" : "hover:bg-gray-100"}`}
                    title="Upload image">📷</button>

            {!recording ? (
                    <button type="button" onClick={startRecording} disabled={loading || transcribing}
                      className={`p-1.5 rounded-lg text-lg transition-colors ${hasVoiceAttachment ? "bg-green-100" : "hover:bg-green-50"}`}
                      title="Record voice">🎤</button>
            ) : (
              <button type="button" onClick={stopRecording}
                      className="p-1.5 bg-red-50 rounded-lg text-lg animate-pulse" title="Stop">⏹️</button>
            )}

            {loading && searchCancelable ? (
              <button
                type="button"
                onClick={onCancelSearch}
                className="inline-flex items-center gap-1.5 px-3 py-2 bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors text-sm font-medium whitespace-nowrap"
                aria-label="Stop search"
              >
                <Square size={14} strokeWidth={2.5} aria-hidden="true" />
                Stop
              </button>
            ) : (
              <button type="submit"
                disabled={loading || recording || transcribing || (!query.trim() && imageFiles.length === 0 && !voiceText.trim())}
                className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-40 transition-colors text-sm font-medium whitespace-nowrap">
                {loading || transcribing ? (
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin inline-block" />
                    {transcribing ? "Transcribing..." : "..."}
                  </span>
                ) : "Search"}
              </button>
            )}
          </div>

          {recording && (
            <div className="px-3 pb-2 flex items-center gap-2 text-xs text-red-500">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse inline-block" />
              Recording... click ⏹️ to stop
            </div>
          )}

          {!recording && (
            <div className="px-3 pb-2 flex gap-3 text-xs text-gray-400">
              <span>📝 Type</span><span>📷 Image</span><span>🎤 Voice</span>
              <span className="text-violet-400">✨ Combine!</span>
            </div>
          )}
        </div>
      </form>
    </div>
  );
}