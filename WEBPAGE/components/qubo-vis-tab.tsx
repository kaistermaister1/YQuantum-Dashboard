"use client";

type QuboVisLightboxProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function QuboVisLightbox({ isOpen, onClose }: QuboVisLightboxProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="overlay overlay--plot" role="presentation" onClick={onClose}>
      <div
        className="modal modal--plot"
        role="dialog"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
        style={{ maxWidth: "90vw", width: "1200px", height: "85vh", padding: "0", display: "flex", flexDirection: "column", overflow: "hidden" }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 20px", borderBottom: "1px solid var(--border)", background: "var(--white)" }}>
          <h2 style={{ margin: 0, fontSize: "1.25rem", color: "var(--yale-blue)", letterSpacing: "-0.02em" }}>QUBO Visualizer</h2>
          <button className="button button-soft" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <div style={{ flex: 1, background: "#000", position: "relative" }}>
          <div style={{
            position: "absolute",
            bottom: "20px",
            left: "20px",
            background: "rgba(0, 0, 0, 0.75)",
            color: "white",
            padding: "16px",
            borderRadius: "8px",
            fontFamily: "var(--font-sans, sans-serif)",
            fontSize: "0.875rem",
            pointerEvents: "none",
            zIndex: 10,
            border: "1px solid rgba(255, 255, 255, 0.15)",
            backdropFilter: "blur(4px)"
          }}>
            <div style={{ fontWeight: 600, marginBottom: "12px", color: "var(--white, #fff)", fontSize: "1rem" }}>Controls</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span><kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)" }}>W</kbd> <kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)" }}>A</kbd> <kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)" }}>S</kbd> <kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)" }}>D</kbd></span>
                <span>to move</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span><kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)" }}>Scroll</kbd></span>
                <span>to zoom</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px", color: "var(--travelers-red, #E31837)", fontWeight: 500 }}>
                <span><kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)", color: "white" }}>←</kbd> <kbd style={{ background: "rgba(255,255,255,0.2)", padding: "2px 6px", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.3)", color: "white" }}>→</kbd></span>
                <span>Arrow keys to move through movie</span>
              </div>
            </div>
          </div>
          <iframe
            src="/qubo_vis/index.html"
            style={{
              width: "100%",
              height: "100%",
              border: "none",
              display: "block"
            }}
            title="QUBO Visualizer"
          />
        </div>
      </div>
    </div>
  );
}
