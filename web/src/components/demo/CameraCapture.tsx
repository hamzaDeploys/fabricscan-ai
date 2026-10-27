import { Camera, SwitchCamera, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

function cameraError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError") return "Camera permission was denied. Allow camera access in the browser's address bar and try again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found on this device.";
  if (name === "NotReadableError") return "The camera is being used by another app. Close it and try again.";
  return "The camera could not be started.";
}

/** Live camera preview with a square guide (the analysis uses the centre square). */
export function CameraCapture({ onClose, onCapture }: { onClose: () => void; onCapture: (blob: Blob) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const captureButton = useRef<HTMLButtonElement>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    setReady(false);
    setError(null);
    (async () => {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        setError("The camera only works on a secure page (https:// or localhost).");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        if (cancelled || !video.current) return;
        video.current.srcObject = stream;
        await video.current.play();
        setReady(true);
        captureButton.current?.focus();
      } catch (err) {
        if (!cancelled) setError(cameraError(err));
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [facing]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const capture = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    c.toBlob((blob) => {
      if (blob) {
        onCapture(blob);
        onClose();
      }
    }, "image/jpeg", 0.92);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" role="dialog" aria-modal="true" aria-labelledby="camera-title">
      <div className="card w-full max-w-2xl overflow-hidden">
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <h3 id="camera-title" className="font-display text-lg font-semibold text-ink">
            Capture a fabric photo
          </h3>
          <button type="button" className="btn btn-ghost px-2" onClick={onClose} aria-label="Close camera">
            <X className="size-5" />
          </button>
        </div>
        <div className="relative aspect-video bg-black">
          <video ref={video} playsInline muted className="size-full object-contain" />
          {ready && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="aspect-square h-[88%] rounded-xl border-2 border-white/90 shadow-[0_0_0_9999px_rgba(0,0,0,0.35)]" />
            </div>
          )}
          {error && <p className="absolute inset-0 flex items-center justify-center p-6 text-center text-white">{error}</p>}
          {!ready && !error && <p className="absolute inset-0 flex items-center justify-center text-white/80">Starting camera…</p>}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
          <p className="text-sm text-muted">Fill the square with fabric. Hold steady, no flash.</p>
          <div className="flex gap-2">
            <button type="button" className="btn btn-secondary" onClick={() => setFacing((f) => (f === "environment" ? "user" : "environment"))}>
              <SwitchCamera className="size-4" aria-hidden="true" /> Switch
            </button>
            <button ref={captureButton} type="button" className="btn btn-primary" onClick={capture} disabled={!ready}>
              <Camera className="size-4" aria-hidden="true" /> Capture
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
