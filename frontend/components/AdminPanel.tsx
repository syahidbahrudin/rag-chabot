"use client";
import React from "react";
import { apiIngest, apiMetrics } from "../lib/api";

export default function AdminPanel() {
  const [metrics, setMetrics] = React.useState<any>(null);
  const [busy, setBusy] = React.useState(false);

  const refresh = async () => {
    const m = await apiMetrics();
    setMetrics(m);
  };

  const ingest = async () => {
    setBusy(true);
    try {
      await apiIngest();
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  React.useEffect(() => {
    refresh();
  }, []);

  return (
    <div style={{
      position: "fixed",
      top: 16,
      right: 16,
      zIndex: 1000,
      minWidth: 280,
      maxWidth: 400,
    }}>
      <div className="card" style={{
        background: "#000000",
        border: "1px solid #2a2a2a",
        padding: 12,
      }}>
        <h2 style={{ 
          marginTop: 0, 
          marginBottom: 12,
          fontSize: 16,
          color: "#ffffff"
        }}>
          Admin
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 8 }}>
          <button
            onClick={ingest}
            disabled={busy}
            style={{
              padding: "10px 12px",
              borderRadius: 6,
              border: "1px solid #2a2a2a",
              background: busy ? "#1a1a1a" : "#2a2a2a",
              color: "#ffffff",
              cursor: busy ? "not-allowed" : "pointer",
              fontWeight: 500,
              fontSize: 13,
              transition: "background 0.2s",
            }}
            onMouseEnter={(e) => {
              if (!busy) {
                e.currentTarget.style.background = "#3a3a3a";
              }
            }}
            onMouseLeave={(e) => {
              if (!busy) {
                e.currentTarget.style.background = "#2a2a2a";
              }
            }}
          >
            {busy ? "Indexing..." : "Ingest sample docs"}
          </button>
          <button
            onClick={refresh}
            style={{
              padding: "10px 12px",
              borderRadius: 6,
              border: "1px solid #2a2a2a",
              background: "#2a2a2a",
              color: "#ffffff",
              cursor: "pointer",
              fontWeight: 500,
              fontSize: 13,
              transition: "background 0.2s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "#3a3a3a";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "#2a2a2a";
            }}
          >
            Refresh metrics
          </button>
        </div>
        {metrics && (
          <div className="code" style={{
            maxHeight: 200,
            overflowY: "auto",
            fontSize: 11,
          }}>
            <pre style={{ margin: 0, color: "#e5e5e5" }}>{JSON.stringify(metrics, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
