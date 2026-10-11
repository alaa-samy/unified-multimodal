"use client";

import { useEffect, useState, useRef } from "react";
import { X } from "lucide-react";

interface ARViewerProps {
  productId: string;
  onClose: () => void;
}

export function ARViewer({ productId, onClose }: ARViewerProps) {
  const [product, setProduct] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const fetchProduct = async () => {
      try {
        setLoading(true);
        setError("");

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        console.log(`🔍 Fetching product ${productId} from ${apiUrl}`);

        const response = await fetch(`${apiUrl}/products/${productId}`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: Failed to fetch product`);
        }

        const foundProduct = await response.json();
        console.log("📦 Product data:", foundProduct);

        if (!foundProduct.model_url) {
          console.error("❌ No model_url in product data");
          throw new Error("3D model not available for this product");
        }

        console.log("✅ Model URL:", foundProduct.model_url);

        setProduct(foundProduct);
      } catch (err: any) {
        console.error("❌ Failed to load product:", err);
        setError(err.message || "Failed to load AR experience");
      } finally {
        setLoading(false);
      }
    };

    fetchProduct();
  }, [productId]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4 text-center">
          <div className="w-16 h-16 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Loading AR Experience
          </h3>
          <p className="text-gray-600">Preparing 3D model...</p>
        </div>
      </div>
    );
  }

  if (error || !product) {
    return (
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4">
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-4xl">⚠️</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              AR Not Available
            </h3>
            <p className="text-gray-600 mb-4">{error || "Failed to load AR experience"}</p>
            {product && product.model_url && (
              <p className="text-xs text-gray-400 break-all">
                Model: {product.model_url}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-full px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/80 to-transparent p-4 z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="text-white">
            <h2 className="text-xl font-semibold">{product.name}</h2>
            <p className="text-sm text-gray-300">{product.subCategory}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition"
            aria-label="Close"
          >
            <X className="w-6 h-6 text-white" />
          </button>
        </div>
      </div>

      {/* 3D Viewer */}
      <div className="w-full h-full flex items-center justify-center p-4 pt-20 pb-32">
        <div className="w-full h-full max-w-5xl">
          <ModelViewer
            modelUrl={product.model_url}
            posterUrl={product.image_url}
            productName={product.name}
          />
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6 z-10">
        <div className="max-w-7xl mx-auto">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 text-white">
            <p className="text-sm text-center mb-2 font-medium">💡 How to use AR</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="flex items-start gap-2">
                <span className="text-purple-400">🖱️</span>
                <span>Click and drag to rotate</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-purple-400">🔍</span>
                <span>Scroll to zoom in/out</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-purple-400">📱</span>
                <span>Tap "View in AR" to see in your space</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ModelViewerProps {
  modelUrl: string;
  posterUrl?: string;
  productName: string;
}

const ModelViewerTag: any = "model-viewer";

function ModelViewer({ modelUrl, posterUrl, productName }: ModelViewerProps) {
  const [modelError, setModelError] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const scriptId = "model-viewer-script";
    const existing = document.getElementById(scriptId) as HTMLScriptElement | null;
    const script = existing || document.createElement("script");
    script.id = scriptId;
    script.type = "module";
    script.src = "https://ajax.googleapis.com/ajax/libs/model-viewer/4.1.0/model-viewer.min.js";

    const markReady = () => {
      if (!cancelled) setReady(true);
    };
    script.addEventListener("load", markReady, { once: true });
    if (!existing) document.head.appendChild(script);
    if (customElements.get("model-viewer")) markReady();
    else customElements.whenDefined("model-viewer").then(markReady).catch(() => {
      if (!cancelled) setModelError(true);
    });

    return () => {
      cancelled = true;
      script.removeEventListener("load", markReady);
    };
  }, []);

  if (modelError) {
    return <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-xl text-white p-8 text-center">
      <div><div className="text-5xl mb-3">⚠️</div><h3 className="text-xl font-semibold mb-2">3D model could not load</h3><p className="text-gray-400 text-sm break-all">{modelUrl}</p></div>
    </div>;
  }

  return (
    <div className="w-full h-full relative rounded-xl overflow-hidden bg-gray-900">
      {(!ready || modelLoading) && <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900 text-white">
        <div className="text-center"><div className="w-12 h-12 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" /><p>{ready ? "Loading 3D model…" : "Loading AR viewer…"}</p></div>
      </div>}
      {ready && (
        <ModelViewerTag
          src={modelUrl}
          poster={posterUrl || undefined}
          alt={productName}
          ar
          ar-modes="webxr scene-viewer quick-look"
          camera-controls
          auto-rotate
          shadow-intensity="1"
          environment-image="neutral"
          exposure="1.1"
          style={{ width: "100%", height: "100%", backgroundColor: "#111827", borderRadius: "0.75rem" }}
          onLoad={() => setModelLoading(false)}
          onError={() => { setModelLoading(false); setModelError(true); }}
        >
          <button slot="ar-button" className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-xl bg-violet-600 px-5 py-3 font-semibold text-white shadow-lg hover:bg-violet-700">
            View in your space
          </button>
        </ModelViewerTag>
      )}
    </div>
  );
}
