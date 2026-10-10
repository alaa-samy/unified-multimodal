"use client";

import { useState, useEffect, useRef } from "react";
import { SearchCard } from "./SearchCard";
import { ARViewer } from "./ARViewer";
import { UnifiedSearchInput } from "./UnifiedSearchInput";
import { ImageGenerator } from "./ImageGenerator";
import { CartDrawer } from "./CartDrawer";
import { OrderConfirmation } from "./OrderConfirmation";
import { useCart } from "@/hooks/useCart";
import { CheckoutPage } from "@/app/checkout/page";

// ── Types ──────────────────────────────────────────────────
interface SearchSession {
  id: string;
  timestamp: number;
  query: string;
  mode: string;
  resultCount: number;
  topScore: number;
  timeMs: number;
  relevant: boolean | null;
  addedToCart: boolean;
  colorFilter: string | null;
  voiceQuery: string | null;
  voiceError: string | null;
  modalities: string[];
}

interface ChatMessage {
  type: "user" | "ai" | "results";
  text?: string;
  hasImage?: boolean;
  imagePreviews?: string[];   // CHANGED: single preview -> array of previews
  voiceQuery?: string;
  voiceError?: string;
  voiceRejectedText?: string;
  thinking?: boolean;
  animate?: boolean;
  results?: any[];
  color?: string | null;
  material?: string | null;   // NEW: leather / velvet / wood ...
  pattern?: string | null;    // NEW: printed / striped / floral ...
  mode?: string;
  lowConfidence?: boolean;
  totalResults?: number;
}

type Screen = "search" | "checkout" | "confirmation";


// ── Typewriter ────────────────────────────────────────────
function useTypewriter(text: string, speed = 22) {
  const [displayed, setDisplayed] = useState("");
  const key = useRef(0);

  useEffect(() => {
    const myKey = ++key.current;
    setDisplayed("");
    if (!text) return;

    const words = text.split(" ");
    let i = 0;
    const t = setInterval(() => {
      if (key.current !== myKey) {
        clearInterval(t);
        return;
      }
      if (i < words.length) {
        setDisplayed((p) => (p ? p + " " + words[i] : words[i]));
        i++;
      } else {
        clearInterval(t);
      }
    }, speed);

    return () => clearInterval(t);
  }, [text]);

  return displayed;
}

function Dots() {
  return (
    <span className="inline-flex gap-1 ml-1 align-middle">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-violet-400 inline-block"
          style={{ animation: `dp 1.2s ease-in-out ${i * 0.2}s infinite` }}
        />
      ))}
      <style>{`@keyframes dp{0%,80%,100%{transform:scale(0.5);opacity:0.3}40%{transform:scale(1);opacity:1}}`}</style>
    </span>
  );
}

function safeChatValue(value: unknown): string {
  if (typeof value !== "string") return "";
  return value
    .replace(/\b(?:undefined|null)\b/gi, "")
    .replace(/\s+([.,;:!?])/g, "$1")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function generateAISummary(data: {
  query: string;
  voiceText: string;
  voiceQuery: unknown;
  voiceError: unknown;
  voiceRejectedText: unknown;
  results: any[];
  totalResults: number;
  lowConfidence: boolean;
  colorFilter: unknown;
  fallbackTier: unknown;
  hasImage: boolean;
}): string {
  const query = safeChatValue(data.query);
  const voiceQuery = safeChatValue(data.voiceText) || safeChatValue(data.voiceQuery);
  const voiceError = safeChatValue(data.voiceError);
  const rejectedText = safeChatValue(data.voiceRejectedText);
  const color = safeChatValue(data.colorFilter);
  const fallbackTier = safeChatValue(data.fallbackTier);
  const originalQuery = [query, voiceQuery].filter(Boolean).join(" ") || (data.hasImage ? "your photo" : "your search");
  const displayQuery = originalQuery;
  const count = Number.isFinite(data.totalResults) ? data.totalResults : data.results.length;
  const description = displayQuery === "your photo" ? "your photo" : `"${displayQuery}"`;

  let summary: string;
  if (data.lowConfidence && count > 0) {
    summary = `I couldn't find a strong match for ${description}, so here are the closest visual matches. Try rephrasing or uploading a photo.`;
  } else if (fallbackTier === "color_relaxed" && count > 0) {
    summary = color
      ? `I found few items matching ${description} in ${color}, so here are the closest visual matches.`
      : `I found few items matching ${description}, so here are the closest visual matches.`;
  } else if (count === 0) {
    summary = `I couldn't find any matches for ${description}. Try changing the color or rephrasing your search.`;
  } else {
    summary = `Here are ${count} closest visual matches for ${description}.`;
  }

  if (voiceError || rejectedText) {
    const voiceNote = voiceError || `I heard "${rejectedText}", but couldn't use that voice input`;
    summary = `${summary.replace(/[.!?]$/, "")}. ${voiceNote}.`;
  }

  return safeChatValue(summary);
}

// ── Chat Bubbles ───────────────────────────────────────────
function UserBubble({
  text,
  hasImage,
  imagePreviews,
  voiceQuery,
  voiceError,
  voiceRejectedText,
}: {
  text?: string;
  hasImage?: boolean;
  imagePreviews?: string[];
  voiceQuery?: string;
  voiceError?: string;
  voiceRejectedText?: string;
}) {
  return (
    <div className="flex gap-3 items-start justify-end">
      <div className="max-w-sm w-full">
        <div className="bg-gradient-to-br from-violet-600 to-indigo-600 text-white rounded-2xl rounded-tr-none px-4 py-3 shadow text-sm leading-relaxed">
          {safeChatValue(text) && <p>{safeChatValue(text)}</p>}
          {imagePreviews && imagePreviews.length > 0 && (
            <div className="mt-2 grid grid-cols-2 gap-2">
              {imagePreviews.map((src, i) => (
                <img
                  key={i}
                  src={src}
                  alt={`Attached ${i + 1}`}
                  className="rounded-xl max-h-48 max-w-full object-cover border-2 border-white/30"
                />
              ))}
            </div>
          )}
          {hasImage && (!imagePreviews || imagePreviews.length === 0) && (
            <p className="opacity-75 text-xs mt-1">📷 Image attached</p>
          )}
          {voiceQuery && (
            <p className="opacity-80 text-xs mt-1">🎤 "{safeChatValue(voiceQuery)}"</p>
          )}
          {(voiceError || voiceRejectedText) && (
            <p className="opacity-80 text-xs mt-1">🎤 {safeChatValue(voiceError) || "Voice input was rejected"}{voiceRejectedText ? ` (heard "${safeChatValue(voiceRejectedText)}")` : ""}</p>
          )}
        </div>
      </div>
      <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 text-xs font-bold flex-shrink-0">
        U
      </div>
    </div>
  );
}

function AIBubble({
  text,
  thinking,
  animate,
}: {
  text: string;
  thinking?: boolean;
  animate?: boolean;
}) {
  const typed = useTypewriter(animate ? text : "", 20);

  return (
    <div className="flex gap-3 items-start">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-indigo-500 flex items-center justify-center text-white text-xs font-bold flex-shrink-0 shadow">
        AI
      </div>
      <div className="flex-1 bg-white rounded-2xl rounded-tl-none px-4 py-3 shadow-sm border border-gray-100 text-gray-700 text-sm leading-relaxed max-w-lg">
        {thinking ? (
          <span className="text-gray-400 italic text-xs">
            Searching<Dots />
          </span>
        ) : (
          <span>{safeChatValue(animate ? typed : text)}</span>
        )}
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────
export default function SearchInterface() {
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeAR, setActiveAR] = useState<string | null>(null);
  const [cartOpen, setCartOpen] = useState(false);
  const [screen, setScreen] = useState<Screen>("search");
  const [orderId, setOrderId] = useState("");
  const [showGenerator, setShowGenerator] = useState(false);
  const [searchCancelable, setSearchCancelable] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const searchControllerRef = useRef<AbortController | null>(null);
  const { totalItems, addItem, sessionId } = useCart();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  const push = (e: ChatMessage) => setChat((p) => [...p, e]);
  const patch = (u: Partial<ChatMessage>) =>
    setChat((p) => {
      const c = [...p];
      c[c.length - 1] = { ...c[c.length - 1], ...u };
      return c;
    });

  async function handleSearch(data: {
    query: string;
    image?: File | null;
    images?: File[];
    voiceText: string;
  }) {
    const query = safeChatValue(data.query);
    const voiceText = safeChatValue(data.voiceText);
    const images = Array.isArray(data.images)
      ? data.images
      : data.image ? [data.image] : [];
    if (!query && images.length === 0 && !voiceText) return;

    setLoading(true);
    const startTime = Date.now();
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // CHANGED: build a preview for every attached image
    const imagePreviews: string[] = [];
    await Promise.all(
      images.map(
        (img) =>
          new Promise<void>((res) => {
            const r = new FileReader();
            r.onload = (e) => {
              if (e.target?.result) imagePreviews.push(e.target.result as string);
              res();
            };
            r.onerror = () => res();
            r.readAsDataURL(img);
          })
      )
    );

    push({
      type: "user",
      text: query,
      hasImage: images.length > 0,
      imagePreviews,
      voiceQuery: voiceText,
    });
    push({ type: "ai", thinking: true });
    const controller = new AbortController();
    searchControllerRef.current = controller;
    setSearchCancelable(true);

    try {
      const form = new FormData();
      form.append("top_k", "20");
      form.append("query", query || "");
      form.append("session_id", sessionId);
      // CHANGED: append EVERY image under the repeated "files" field
      images.forEach((img) => form.append("files", img));
      if (voiceText) form.append("voice_text", voiceText);

      const res = await fetch(`${apiUrl}/search/intelligent`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`Backend error ${res.status}`);

      const d = await res.json();
      const results = Array.isArray(d?.accurate_results) ? d.accurate_results : [];
      const totalResults = typeof d?.total_results === "number" && Number.isFinite(d.total_results)
        ? d.total_results
        : results.length;
      const voiceError = safeChatValue(d?.voice_error);
      const voiceRejectedText = safeChatValue(d?.voice_rejected_text);
      const lowConfidence = d?.low_confidence === true;
      const colorFilter = safeChatValue(d?.color_filter) || null;
      const searchMode = safeChatValue(d?.search_mode);
      const responseTime = Date.now() - startTime;

      // Keep the confirmed transcript in the user bubble and add backend voice diagnostics.
      setChat((p) => {
        const c = [...p];
        const lastIdx = c.length - 2;
        if (lastIdx >= 0 && c[lastIdx].type === "user") {
          c[lastIdx] = { ...c[lastIdx], voiceError, voiceRejectedText };
        }
        return c;
      });

      const summary = generateAISummary({
        query,
        voiceText,
        voiceQuery: d?.voice_query,
        voiceError: d?.voice_error,
        voiceRejectedText: d?.voice_rejected_text,
        results,
        totalResults,
        lowConfidence,
        colorFilter: d?.color_filter,
        fallbackTier: d?.fallback_tier,
        hasImage: imagePreviews.length > 0,
      });

      patch({
        thinking: false,
        text: summary,
        animate: true,
      });

      if (results.length > 0) {
        push({
          type: "results",
          results,
          color: colorFilter,
          material: safeChatValue(d?.material_filter) || null,   // NEW
          pattern: safeChatValue(d?.pattern_filter) || null,     // NEW
          mode: searchMode,
          lowConfidence,
          totalResults,
        });
      }

      // Log session for evaluation
      console.log({
        sessionId,
        query,
        voiceQuery: safeChatValue(d?.voice_query) || voiceText,
        voiceError,
        mode: searchMode,
        results: results.length,
        topScore: results[0]?.score || 0,
        responseTime,
      });
    } catch (err) {
      if (controller.signal.aborted) {
        patch({ thinking: false, text: "Search stopped.", animate: false });
        return;
      }
      const message = err instanceof Error ? err.message : "Unknown error";
      patch({
        thinking: false,
        text: message.toLowerCase().includes("failed to fetch")
          ? `I can't reach the search server at ${apiUrl}. Please check that the backend is running.`
          : `Something went wrong: ${safeChatValue(message) || "Unknown error"}.`,
        animate: true,
      });
    } finally {
      if (searchControllerRef.current === controller) searchControllerRef.current = null;
      setSearchCancelable(false);
      setLoading(false);
    }
  }

  const handleGenerateComplete = async (imageUrl: string, prompt: string) => {
    setShowGenerator(false);
    setLoading(true);
    patch({
      thinking: true,
      text: "Nice design! Finding similar items in our catalog...",
    });

    try {
      const res = await fetch(imageUrl);
      if (!res.ok) throw new Error(`Generated image could not be loaded (${res.status})`);
      const blob = await res.blob();
      const file = new File([blob], `generated_${Date.now()}.jpg`, { type: blob.type });

      // Add the AI message showing what was generated
      push({
        type: "ai",
        text: `Here is the furniture I generated to match your idea: "${safeChatValue(prompt)}"`,
        animate: true,
      });

      // Inject the generated image in the chat flow
      push({
        type: "user",
        text: `(Use this generated image for search)`,
        hasImage: true,
        imagePreviews: [imageUrl],
      });

      // Pass it directly to handleSearch (as a single-element array now)
      handleSearch({ query: "", images: [file], voiceText: "" });
    } catch (e) {
      console.error(e);
      const message = e instanceof Error ? e.message : "Unknown error";
      push({
        type: "ai",
        text: `I couldn't load the generated image. ${safeChatValue(message) || "Please try generating it again."}`,
        animate: true,
      });
      setLoading(false);
    }
  };

  if (screen === "checkout") {
    return (
      <CheckoutPage
        onBack={() => setScreen("search")}
        onSuccess={(id) => {
          setOrderId(id);
          setScreen("confirmation");
        }}
      />
    );
  }

  if (screen === "confirmation") {
    return (
      <OrderConfirmation
        orderId={orderId}
        onContinue={() => {
          setChat([]);
          setScreen("search");
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-violet-50/20 to-indigo-50/40 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-8 pb-2 flex-shrink-0">
        <div className="flex-1 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-100 text-violet-700 text-xs font-semibold mb-2 border border-violet-200 tracking-wide uppercase">
            <span className="w-1.5 h-1.5 bg-violet-500 rounded-full animate-pulse" />
            AI Furniture Search
          </div>
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 tracking-tight">
            What are you looking for?
          </h1>
          <p className="text-gray-400 mt-1 text-sm">
            Describe it · show it · say it
          </p>
        </div>
        <button
          onClick={() => setCartOpen(true)}
          className="relative flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-xl shadow-sm hover:border-violet-300 transition-all text-sm font-medium text-gray-700 flex-shrink-0"
        >
          🛒 Cart
          {totalItems > 0 && (
            <span className="absolute -top-2 -right-2 w-5 h-5 bg-violet-600 text-white text-xs rounded-full flex items-center justify-center font-bold">
              {totalItems}
            </span>
          )}
        </button>
      </div>

      {/* Chat Area */}
      <div className="flex-1 max-w-3xl w-full mx-auto px-4 pb-6 space-y-5">
        {chat.length === 0 && (
          <AIBubble
            text="Hi! I can find furniture for you using text, images, or voice — or all three at once. Try something like 'black leather sofa' or upload a photo of a style you like."
            animate={false}
          />
        )}

        {chat.map((entry, i) => {
          if (entry.type === "user") {
            return (
              <UserBubble
                key={i}
                text={entry.text}
                hasImage={entry.hasImage}
                imagePreviews={entry.imagePreviews}
                voiceQuery={entry.voiceQuery}
                voiceError={entry.voiceError}
                voiceRejectedText={entry.voiceRejectedText}
              />
            );
          }

          if (entry.type === "ai") {
            return (
              <AIBubble
                key={i}
                text={entry.text || ""}
                thinking={entry.thinking}
                animate={entry.animate}
              />
            );
          }

          if (entry.type === "results" && entry.results) {
            return (
              <div key={i} className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">
                    {entry.totalResults ?? entry.results.length} Results
                  </span>
                  <div className="flex-1 h-px bg-gray-200" />
                  {entry.color && (
                    <span className="flex items-center gap-1.5 text-xs text-violet-700 font-medium">
                      <span
                        className="w-3 h-3 rounded-full border border-violet-300 inline-block"
                        style={{
                          background:
                            entry.color === "gray"
                              ? "#9ca3af"
                              : entry.color === "navy"
                                ? "#1e3a5f"
                                : entry.color,
                        }}
                      />
                      {entry.color}
                    </span>
                  )}
                  {entry.material && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-medium">
                      {entry.material}
                    </span>
                  )}
                  {entry.pattern && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-pink-100 text-pink-700 font-medium">
                      {entry.pattern}
                    </span>
                  )}
                  {entry.mode && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-600 font-medium">
                      {entry.mode}
                    </span>
                  )}
                </div>
                {entry.lowConfidence && (
                  <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
                    No strong match - showing the closest items
                  </p>
                )}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {entry.results.slice(0, 12).map((p) => (
                    <SearchCard
                      key={p.id}
                      id={p.id}
                      name={p.name}
                      category={p.category}
                      subCategory={p.subCategory}
                      image_url={p.image_url}
                      score={p.score}
                      color={p.color}
                      has_3d_model={p.has_3d_model}
                      model_url={p.model_url}
                      onViewAR={(id: any) => setActiveAR(String(id))}
                      onAddToCart={() =>
                        addItem({
                          id: Number(p.id),
                          name: p.name,
                          category: p.category,
                          color: p.color,
                          image_url: p.image_url,
                          score: p.score,
                        })
                      }
                    />
                  ))}
                </div>
              </div>
            );
          }

          return null;
        })}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="sticky bottom-0 bg-white/80 backdrop-blur-md border-t border-gray-200 px-4 py-3 flex-shrink-0">
        <div className="max-w-3xl mx-auto flex flex-col items-center">
          <UnifiedSearchInput
            onSearch={handleSearch}
            loading={loading}
            searchCancelable={searchCancelable}
            onCancelSearch={() => searchControllerRef.current?.abort()}
          />
          <button
            onClick={() => setShowGenerator(true)}
            className="mt-3 text-xs font-semibold text-violet-600 hover:text-violet-800 transition-colors flex items-center gap-1"
          >
            ✨ Having trouble finding it? Generate it with AI!
          </button>
        </div>
      </div>

      {/* Modals */}
      {showGenerator && (
        <ImageGenerator
          onGenerateComplete={handleGenerateComplete}
          onCancel={() => setShowGenerator(false)}
        />
      )}
      <CartDrawer
        open={cartOpen}
        onClose={() => setCartOpen(false)}
        onCheckout={() => {
          setCartOpen(false);
          setScreen("checkout");
        }}
      />
      {activeAR && (
        <ARViewer productId={activeAR} onClose={() => setActiveAR(null)} />
      )}
    </div>
  );
}