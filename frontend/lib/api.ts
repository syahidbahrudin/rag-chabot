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
  let response: Response;

  try {
    response = await fetch(`${API_BASE}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k }),
    });
  } catch (error) {
    // Handle network errors (connection refused, timeout, etc.)
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        "Network error: Could not connect to the server. Please check if the backend is running."
      );
    }
    throw new Error(
      `Network error: ${
        error instanceof Error ? error.message : "Unknown error"
      }`
    );
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new Error(
      `Request failed (${response.status}): ${errorText || "Stream failed"}`
    );
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error("No reader available");
  }

  let buffer = "";

  try {
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
  } catch (error) {
    throw new Error(
      `Streaming error: ${
        error instanceof Error ? error.message : "Unknown error"
      }`
    );
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
