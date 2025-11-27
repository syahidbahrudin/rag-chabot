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
  const [loading, setLoading] = React.useState(false);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const send = async () => {
    if (!q.trim()) return;
    const my = { role: "user" as const, content: q };
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
        q,
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
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#343541] overflow-hidden">
      {/* Messages Container */}
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
              className={`group flex gap-4 mb-6 ${
                m.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {/* Message Content */}
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

                {m.citations && m.citations.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {m.citations.map((c, idx) => (
                      <span
                        key={idx}
                        className="bg-[#2a2a2a] text-gray-300 px-2 py-1 rounded-full text-xs border border-[#3a3a3a]"
                        title={c.section || ""}
                      >
                        {c.title}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex gap-4 mb-6 justify-start">
              {/* <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[#19c37d] flex items-center justify-center text-white font-semibold text-sm">
                  AI
                </div> */}
              <div className="text-gray-400 italic">Thinking...</div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Bar */}
      <div className="">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <div className="relative flex items-center bg-[#40414f] rounded-2xl border border-[#565869] shadow-lg">
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
              className="flex-1 bg-transparent text-white placeholder-gray-400 ml-3 px-2 py-3 outline-none resize-none"
            />
            <button>{"->"}</button>
          </div>
          <p className="text-xs text-gray-500 text-center mt-2">
            AI can make mistakes. Check important info.
          </p>
        </div>
      </div>
    </div>
  );
}
