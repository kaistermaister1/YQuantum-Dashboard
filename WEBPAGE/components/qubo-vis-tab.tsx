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
        <div style={{ flex: 1, background: "#000" }}>
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
