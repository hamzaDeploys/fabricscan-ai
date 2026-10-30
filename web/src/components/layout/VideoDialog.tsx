import { Download, X } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type MouseEvent, type ReactNode } from "react";

import { DOWNLOADS } from "../../config/site";

const VideoContext = createContext<() => void>(() => undefined);

/** The presentation video in a popup. The player only exists while the popup is open, so the file loads on demand. */
export function VideoProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const openVideo = useCallback(() => setOpen(true), []);

  useEffect(() => {
    if (open) dialog.current?.showModal();
    else dialog.current?.close();
  }, [open]);

  return (
    <VideoContext.Provider value={openVideo}>
      {children}
      <dialog
        ref={dialog}
        className="video-dialog dark"
        aria-labelledby="video-title"
        data-testid="video-dialog"
        onClose={() => setOpen(false)}
        onClick={(e) => e.target === e.currentTarget && setOpen(false)}
      >
        {open && (
          <div>
            <div className="flex items-center justify-between gap-3 border-b border-line py-2 pr-2 pl-4 sm:pl-5">
              <h2 id="video-title" className="truncate font-display text-sm font-semibold text-ink sm:text-base">
                FabricScan AI: project presentation
              </h2>
              <div className="flex shrink-0 items-center gap-1">
                <a href={DOWNLOADS.video} download className="btn btn-ghost px-2.5" aria-label="Download the video" title="Download the video">
                  <Download className="size-5" aria-hidden="true" />
                </a>
                <button type="button" className="btn btn-ghost px-2.5" aria-label="Close video" title="Close" onClick={() => setOpen(false)}>
                  <X className="size-5" aria-hidden="true" />
                </button>
              </div>
            </div>
            <video
              src={DOWNLOADS.video}
              controls
              autoPlay
              playsInline
              data-testid="video-player"
              className="block aspect-video max-h-[calc(100dvh-6rem)] w-full bg-black"
            />
          </div>
        )}
      </dialog>
    </VideoContext.Provider>
  );
}

/** Props for a link that plays the video in the popup. Crawlers and "open in new tab" still get the file itself. */
export function useVideoLink() {
  const openVideo = useContext(VideoContext);
  return {
    href: DOWNLOADS.video,
    onClick: (e: MouseEvent) => {
      e.preventDefault();
      openVideo();
    },
  };
}
