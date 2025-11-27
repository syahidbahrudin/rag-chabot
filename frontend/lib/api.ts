export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function apiAsk(query: string, k: number = 4) {
  const r = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!r.ok) throw new Error("Ask failed");
  return r.json();
}

export async function apiAskStream(
  query: string,
  k: number = 4,
  onChunk: (chunk: string) => void,
  onMetadata?: (metadata: {
    citations: { title: string; section?: string }[];
  }) => void,
  onDone?: () => void
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });

  if (!response.ok) {
    throw new Error("Stream failed");
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error("No reader available");
  }

  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === "chunk" && data.content) {
            onChunk(data.content);
          } else if (data.type === "metadata" && onMetadata) {
            onMetadata(data);
          } else if (data.type === "done" && onDone) {
            onDone();
          }
        } catch (e) {
          console.error("Failed to parse SSE data:", e);
        }
      }
    }
  }
}

export async function apiIngest() {
  const r = await fetch(`${API_BASE}/api/ingest`, { method: "POST" });
  if (!r.ok) throw new Error("Ingest failed");
  return r.json();
}

export async function apiMetrics() {
  const r = await fetch(`${API_BASE}/api/metrics`);
  if (!r.ok) throw new Error("Metrics failed");
  return r.json();
}
