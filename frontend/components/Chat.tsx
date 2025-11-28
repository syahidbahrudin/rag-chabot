"use client";
import React from "react";
import ReactMarkdown from "react-markdown";
import { apiAskStream } from "../lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: { title: string; section?: string }[];
  chunks?: { title: string; section?: string; text: string }[];
};

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [q, setQ] = React.useState("");
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  React.useEffect(() => {
    if (!isSubmitting) return;
    scrollToBottom();
  }, [isSubmitting]);

  const send = async (question?: string) => {
    if (loading) return;
    setIsSubmitting(true);
    const questionToSend = question || q;
    if (!questionToSend.trim()) {
      return;
    }
    const my = { role: "user" as const, content: questionToSend };
    setMessages((m) => [...m, my]);
    setLoading(true);
    // Create a placeholder message for streaming
    const aiMessage: Message = {
      role: "assistant",
      content: "",
      citations: [],
    };
    setMessages((m) => [...m, aiMessage]);

    try {
      await apiAskStream(
        questionToSend,
        4,
        (chunk: string) => {
          // Update the last message with new chunk
          setMessages((m) => {
            const newMessages = [...m];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg.role === "assistant") {
              lastMsg.content += chunk;
            }
            return newMessages;
          });
        },
        (metadata) => {
          // Update citations when metadata arrives - deduplicate by title
          setMessages((m) => {
            const newMessages = [...m];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg.role === "assistant") {
              // Deduplicate citations by title
              const seen = new Set<string>();
              lastMsg.citations = metadata.citations.filter(
                (c: { title: string; section?: string }) => {
                  if (seen.has(c.title)) {
                    return false;
                  }
                  seen.add(c.title);
                  return true;
                }
              );
            }
            return newMessages;
          });
        },
        () => {
          // Streaming complete
          setLoading(false);
        }
      );
    } catch (e: any) {
      setMessages((m) => {
        const newMessages = [...m];
        const lastMsg = newMessages[newMessages.length - 1];
        if (lastMsg.role === "assistant") {
          lastMsg.content = "Error: " + e.message;
        }
        return newMessages;
      });
      setLoading(false);
    } finally {
      setQ("");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col h-screen  overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-8">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
              <p className="text-lg">Start a conversation</p>
            </div>
          )}

          {messages.map((m, i) => (
            <div
              key={i}
              className={`group flex flex-col gap-2 mb-6 ${
                m.role === "user" ? "items-end" : "items-start"
              }`}
            >
              <div
                className={`${
                  m.role === "user"
                    ? "bg-neutral-800 text-white rounded-2xl px-4 py-3 max-w-[85%]"
                    : "text-[#ececf1] max-w-[85%]"
                }`}
              >
                {m.role === "assistant" ? (
                  <div className="prose prose-invert max-w-none">
                    <ReactMarkdown
                      components={{
                        h1: ({ node, ...props }) => (
                          <h1
                            className="text-white text-2xl font-semibold mt-4 mb-2"
                            {...props}
                          />
                        ),
                        h2: ({ node, ...props }) => (
                          <h2
                            className="text-white text-xl font-semibold mt-4 mb-2"
                            {...props}
                          />
                        ),
                        h3: ({ node, ...props }) => (
                          <h3
                            className="text-white text-lg font-semibold mt-3 mb-2"
                            {...props}
                          />
                        ),
                        p: ({ node, ...props }) => (
                          <p
                            className="mb-3 leading-relaxed text-[#ececf1]"
                            {...props}
                          />
                        ),
                        ul: ({ node, ...props }) => (
                          <ul
                            className="list-disc list-inside mb-3 space-y-1 text-[#ececf1]"
                            {...props}
                          />
                        ),
                        ol: ({ node, ...props }) => (
                          <ol
                            className="list-decimal list-inside mb-3 space-y-1 text-[#ececf1]"
                            {...props}
                          />
                        ),
                        li: ({ node, ...props }) => (
                          <li className="mb-1 text-[#ececf1]" {...props} />
                        ),
                        strong: ({ node, ...props }) => (
                          <strong
                            className="text-white font-semibold"
                            {...props}
                          />
                        ),
                        code: ({
                          node,
                          className,
                          children,
                          ...props
                        }: any) => {
                          const isInline = !className;
                          return isInline ? (
                            <code
                              className="bg-[#2a2a2a] px-1.5 py-0.5 rounded text-sm text-[#88ccff] font-mono"
                              {...props}
                            >
                              {children}
                            </code>
                          ) : (
                            <code
                              className="block bg-[#1a1a1a] p-3 rounded-lg text-sm text-gray-300 font-mono overflow-x-auto mb-3 border border-[#2a2a2a]"
                              {...props}
                            >
                              {children}
                            </code>
                          );
                        },
                        blockquote: ({ node, ...props }) => (
                          <blockquote
                            className="border-l-4 border-[#4a4a4a] pl-4 my-3 text-gray-300 italic"
                            {...props}
                          />
                        ),
                        a: ({ node, ...props }) => (
                          <a className="text-[#88ccff] underline" {...props} />
                        ),
                      }}
                    >
                      {m.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div className="text-white leading-relaxed whitespace-pre-wrap">
                    {m.content}
                  </div>
                )}
              </div>

              {m.role === "assistant" &&
                m.citations &&
                m.citations.length > 0 && (
                  <div className="flex flex-wrap gap-2 max-w-[85%]">
                    {m.citations.map((c, idx) => (
                      <span
                        key={idx}
                        className="bg-[#2a2a2a] text-gray-300 px-3 py-1.5 rounded-full text-xs border border-zinc-700 font-medium"
                        title={c.section || ""}
                      >
                        {c.title}
                      </span>
                    ))}
                  </div>
                )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-4 mb-6 justify-start">
              <div className="text-gray-400 italic">Thinking...</div>
            </div>
          )}

          <div ref={messagesEndRef} className="h-[66dvh]" />
        </div>
      </div>

      <div className="">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <div className=" flex flex-row gap-5 mb-5">
            <button
              className="bg-neutral-800 rounded-2xl border border-zinc-700 shadow-lg p-2 text-white cursor-pointer"
              disabled={loading}
              onClick={() => {
                send("Can a customer return a damaged blender after 20 days?");
              }}
            >
              Can a customer return a damaged blender after 20 days?
            </button>
            <button
              className="bg-neutral-800 rounded-2xl border border-zinc-700 shadow-lg p-2 text-white cursor-pointer"
              disabled={loading}
              onClick={() => {
                send(
                  "What's the shipping SLA to East Malaysia for bulky items?"
                );
              }}
            >
              What's the shipping SLA to East Malaysia for bulky items?
            </button>
          </div>
          <div className="relative flex items-center bg-neutral-800 rounded-2xl border border-zinc-700 shadow-lg">
            <input
              type="text"
              placeholder="Ask anything"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              className="flex-1 bg-transparent text-white placeholder-gray-400 ml-3 px-2 py-4 outline-none resize-none"
            />
            <button
              className="bg-white rounded-xl p-2 mr-3 cursor-pointer"
              onClick={() => send()}
              disabled={loading}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24">
                <path d="M7 7h8.586L5.293 17.293l1.414 1.414L17 8.414V17h2V5H7v2z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
