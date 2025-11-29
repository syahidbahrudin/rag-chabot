"use client";
import React from "react";
import { apiIngest, apiMetrics } from "../../lib/api";
import Link from "next/link";

interface Metrics {
  total_docs: number;
  total_chunks: number;
  avg_retrieval_latency_ms: number;
  avg_generation_latency_ms: number;
  embedding_model: string;
  llm_model: string;
}

interface IngestResult {
  indexed_docs: number;
  indexed_chunks: number;
}

export default function AdminPage() {
  const [metrics, setMetrics] = React.useState<Metrics | null>(null);
  const [ingestResult, setIngestResult] = React.useState<IngestResult | null>(
    null
  );
  const [isIngesting, setIsIngesting] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = React.useState(false);

  const fetchMetrics = async () => {
    try {
      setError(null);
      const m = await apiMetrics();
      setMetrics(m);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch metrics");
      console.error("Error fetching metrics:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleIngest = async () => {
    setIsIngesting(true);
    setError(null);
    setIngestResult(null);
    try {
      const result = await apiIngest();
      setIngestResult(result);
      // Refresh metrics after ingestion
      await fetchMetrics();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to ingest documents"
      );
      console.error("Error ingesting:", err);
    } finally {
      setIsIngesting(false);
    }
  };

  React.useEffect(() => {
    fetchMetrics();
  }, []);

  return (
    <div className="min-h-screen bg-neutral-950 text-white p-8">
      <Link
        className="absolute top-3 right-3 text-white bg-neutral-800 px-4 py-2 rounded-md"
        href="/"
      >
        Back to Chat
      </Link>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Admin Dashboard</h1>
          <p className="text-neutral-400">
            Manage document ingestion and view system metrics
          </p>
        </div>

        {/* Ingest Section */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Document Ingestion</h2>
          <div className="flex items-center gap-4">
            <button
              onClick={handleIngest}
              disabled={isIngesting}
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                isIngesting
                  ? "bg-neutral-800 text-neutral-500 cursor-not-allowed"
                  : "bg-neutral-600 hover:bg-neutral-700 text-white"
              }`}
            >
              {isIngesting ? (
                <span className="flex items-center gap-2">
                  <svg
                    className="animate-spin h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Indexing...
                </span>
              ) : (
                "Ingest Documents"
              )}
            </button>
            {ingestResult && (
              <div className="text-sm text-green-400">
                ✓ Indexed {ingestResult.indexed_docs} documents,{" "}
                {ingestResult.indexed_chunks} chunks
              </div>
            )}
          </div>
        </div>

        {/* Metrics Section */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">System Metrics</h2>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchMetrics}
                disabled={isLoading}
                className="px-4 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-sm font-medium transition-colors"
              >
                {isLoading ? "Loading..." : "Refresh"}
              </button>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-sm">
              Error: {error}
            </div>
          )}

          {isLoading && !metrics ? (
            <div className="text-center py-12 text-neutral-400">
              <svg
                className="animate-spin h-8 w-8 mx-auto mb-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Loading metrics...
            </div>
          ) : metrics ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Total Documents */}
              <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4">
                <div className="text-sm text-neutral-400 mb-1">
                  Total Documents
                </div>
                <div className="text-3xl font-bold text-white">
                  {metrics.total_docs}
                </div>
              </div>

              {/* Total Chunks */}
              <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4">
                <div className="text-sm text-neutral-400 mb-1">
                  Total Chunks
                </div>
                <div className="text-3xl font-bold text-white">
                  {metrics.total_chunks}
                </div>
              </div>

              {/* Average Retrieval Latency */}
              <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4">
                <div className="text-sm text-neutral-400 mb-1">
                  Avg Retrieval Latency
                </div>
                <div className="text-3xl font-bold text-white">
                  {metrics.avg_retrieval_latency_ms.toFixed(2)}{" "}
                  <span className="text-lg text-neutral-400">ms</span>
                </div>
              </div>

              {/* Average Generation Latency */}
              <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4">
                <div className="text-sm text-neutral-400 mb-1">
                  Avg Generation Latency
                </div>
                <div className="text-3xl font-bold text-white">
                  {metrics.avg_generation_latency_ms.toFixed(2)}{" "}
                  <span className="text-lg text-neutral-400">ms</span>
                </div>
              </div>

              {/* Embedding Model */}
              <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4">
                <div className="text-sm text-neutral-400 mb-1">
                  Embedding Model
                </div>
                <div className="text-lg font-medium text-white break-all">
                  {metrics.embedding_model}
                </div>
              </div>

              {/* LLM Model */}
              <div className="bg-neutral-800 border border-neutral-700 rounded-lg p-4">
                <div className="text-sm text-neutral-400 mb-1">LLM Model</div>
                <div className="text-lg font-medium text-white break-all">
                  {metrics.llm_model}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-neutral-400">
              No metrics available
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
