"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { api, ChatSource, Municipality } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Bot,
  User,
  ExternalLink,
  Database,
  FileText,
  Image,
  Globe,
  Search,
  X,
  Sparkles,
  ChevronDown,
  Plus,
  Trash2,
  MessageSquare,
  Clock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Source {
  municipality: string;
  zone_code?: string;
  source_url?: string;
  source_document?: string;
  type?: string;
  sql?: string;
}

interface ThoughtStep {
  name: string;
  description: string;
  detail?: string;
  summary?: string;
  preview?: string;
  done: boolean;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  thoughts?: ThoughtStep[];
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SUGGESTED_QUESTIONS = [
  "Can I build a laneway house in Toronto? What are the rules?",
  "Which zones allow the tallest buildings and what are the height limits?",
  "What's the difference between R1 and RM zones?",
  "Are short-term rentals like Airbnb allowed? In which zones?",
  "What are the parking requirements for a new condo development?",
  "Compare density limits across residential zones",
  "What does the official plan say about affordable housing?",
  "Which zones allow mixed-use (residential + commercial)?",
];

const SOURCE_CONFIG: Record<string, { icon: typeof Database; label: string; className: string }> = {
  sql: { icon: Database, label: "Database", className: "bg-purple-50 border-purple-200 text-purple-700" },
  document_chunk: { icon: FileText, label: "Document", className: "bg-blue-50 border-blue-200 text-blue-700" },
  text: { icon: FileText, label: "Document", className: "bg-blue-50 border-blue-200 text-blue-700" },
  image: { icon: Image, label: "Zoning Map", className: "bg-green-50 border-green-200 text-green-700" },
  pdf_page: { icon: FileText, label: "PDF", className: "bg-orange-50 border-orange-200 text-orange-700" },
  web: { icon: Globe, label: "Web", className: "bg-teal-50 border-teal-200 text-teal-700" },
};

const DEFAULT_SOURCE = { icon: Search, label: "Source", className: "bg-muted border-border text-muted-foreground" };

const TOOL_ICONS: Record<string, typeof Database> = {
  query_database: Database,
  search_knowledge_base: FileText,
  search_web: Globe,
};

const STORAGE_KEY = "zonemap-chat-sessions";
const MAX_SESSIONS = 50;
const MAX_MESSAGES_PER_SESSION = 100;

// ---------------------------------------------------------------------------
// Session persistence
// ---------------------------------------------------------------------------
function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function loadSessions(): ChatSession[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as ChatSession[];
  } catch { /* ignore */ }
  return [];
}

function saveSessions(sessions: ChatSession[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS)));
  } catch { /* ignore */ }
}

function deriveTitle(messages: Message[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return "New Chat";
  const text = firstUser.content.trim();
  return text.length > 40 ? text.slice(0, 40) + "..." : text;
}

const WELCOME_MESSAGE: Message = {
  role: "assistant",
  content:
    "Hi! I'm ZoneMap's AI assistant. Ask me anything about Canadian zoning regulations — I can query the database, search documents, and look up the latest info.",
};

// ---------------------------------------------------------------------------
// ChatPanel
// ---------------------------------------------------------------------------
interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  initialQuestion?: string;
  onQuestionConsumed?: () => void;
}

export function ChatPanel({ isOpen, onClose, initialQuestion, onQuestionConsumed }: ChatPanelProps) {
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    const loaded = loadSessions();
    if (loaded.length > 0) return loaded;
    const first: ChatSession = {
      id: generateId(),
      title: "New Chat",
      messages: [WELCOME_MESSAGE],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    return [first];
  });
  const [activeId, setActiveId] = useState<string>(() => {
    const loaded = loadSessions();
    return loaded.length > 0 ? loaded[0].id : sessions[0]?.id || "";
  });
  const [showHistory, setShowHistory] = useState(false);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentThoughts, setCurrentThoughts] = useState<ThoughtStep[]>([]);
  const [streamingContent, setStreamingContent] = useState("");
  const [municipalities, setMunicipalities] = useState<Municipality[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const consumedQuestionRef = useRef<string | null>(null);

  const activeSession = sessions.find((s) => s.id === activeId);
  const messages = activeSession?.messages || [WELCOME_MESSAGE];

  useEffect(() => {
    api.getMunicipalities().then(setMunicipalities).catch(() => {});
  }, []);

  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, currentThoughts, streamingContent]);

  const updateActiveMessages = useCallback(
    (updater: (prev: Message[]) => Message[]) => {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== activeId) return s;
          const newMessages = updater(s.messages).slice(-MAX_MESSAGES_PER_SESSION);
          return {
            ...s,
            messages: newMessages,
            title: deriveTitle(newMessages),
            updatedAt: Date.now(),
          };
        }),
      );
    },
    [activeId],
  );

  const handleSend = useCallback(
    async (questionOverride?: string) => {
      const question = (questionOverride || input).trim();
      if (!question || loading) return;

      setInput("");
      updateActiveMessages((prev) => [...prev, { role: "user", content: question }]);
      setLoading(true);
      setCurrentThoughts([]);
      setStreamingContent("");

      const historyForApi = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));

      let answer = "";
      let sources: ChatSource[] = [];
      const thoughts: ThoughtStep[] = [];

      try {
        for await (const event of api.chatStream(question, undefined, historyForApi)) {
          switch (event.type) {
            case "tool_start": {
              const step: ThoughtStep = {
                name: event.name,
                description: event.description,
                detail: event.detail,
                done: false,
              };
              thoughts.push(step);
              setCurrentThoughts([...thoughts]);
              break;
            }
            case "tool_end": {
              const last = thoughts.find((t) => t.name === event.name && !t.done);
              if (last) {
                last.summary = event.summary;
                last.preview = event.preview;
                last.done = true;
                setCurrentThoughts([...thoughts]);
              }
              break;
            }
            case "token":
              answer += event.content;
              setStreamingContent(answer);
              break;
            case "done":
              sources = event.sources;
              break;
            case "error":
              answer = `Sorry, I encountered an error: ${event.message}`;
              break;
          }
        }
      } catch {
        if (!answer) {
          answer = "Sorry, I encountered an error. Please try again.";
        }
      }

      updateActiveMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer || "I wasn't able to generate a response. Please try again.",
          sources,
          thoughts: thoughts.length > 0 ? thoughts : undefined,
        },
      ]);
      setLoading(false);
      setCurrentThoughts([]);
      setStreamingContent("");
    },
    [input, loading, messages, updateActiveMessages],
  );

  useEffect(() => {
    if (
      initialQuestion &&
      !loading &&
      consumedQuestionRef.current !== initialQuestion
    ) {
      consumedQuestionRef.current = initialQuestion;
      handleSend(initialQuestion);
      onQuestionConsumed?.();
    }
  }, [initialQuestion]);

  function createNewChat() {
    const newSession: ChatSession = {
      id: generateId(),
      title: "New Chat",
      messages: [WELCOME_MESSAGE],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveId(newSession.id);
    setShowHistory(false);
  }

  function deleteSession(id: string) {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== id);
      if (filtered.length === 0) {
        const fallback: ChatSession = {
          id: generateId(),
          title: "New Chat",
          messages: [WELCOME_MESSAGE],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setActiveId(fallback.id);
        return [fallback];
      }
      if (id === activeId) {
        setActiveId(filtered[0].id);
      }
      return filtered;
    });
  }

  function switchSession(id: string) {
    setActiveId(id);
    setShowHistory(false);
  }

  if (!isOpen) return null;

  const muniNames = municipalities.map((m) => m.name);

  return (
    <div className="flex flex-col h-full bg-background" style={{ minHeight: 0 }}>
      {/* Toolbar */}
      <div className="flex-shrink-0 border-b border-border bg-muted/30 px-4 sm:px-6 py-2">
        <div className="max-w-3xl mx-auto flex items-center gap-2">
          <div className="flex items-center gap-2 mr-auto">
            <div className="w-7 h-7 rounded-lg bg-primary flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary-foreground" />
            </div>
            <h3 className="text-sm font-semibold">AI Assistant</h3>
          </div>

          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs gap-1.5"
            onClick={createNewChat}
          >
            <Plus className="h-3.5 w-3.5" />
            New Chat
          </Button>

          <Button
            variant={showHistory ? "secondary" : "outline"}
            size="sm"
            className="h-8 text-xs gap-1.5"
            onClick={() => setShowHistory((v) => !v)}
          >
            <Clock className="h-3.5 w-3.5" />
            History
          </Button>

          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose} title="Back to app">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* History drawer */}
      {showHistory && (
        <div className="flex-shrink-0 border-b border-border bg-muted/20 animate-fade-in max-h-64 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 sm:px-6 py-2">
            <div className="space-y-0.5">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className={`group flex items-center gap-2 px-3 py-2 rounded-md text-sm cursor-pointer transition-colors ${
                    session.id === activeId
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                  <button
                    onClick={() => switchSession(session.id)}
                    className="flex-1 text-left truncate"
                  >
                    {session.title}
                  </button>
                  <span className="text-[10px] opacity-50 flex-shrink-0">
                    {new Date(session.updatedAt).toLocaleDateString()}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSession(session.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity flex-shrink-0"
                    title="Delete chat"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Cities indicator */}
      {muniNames.length > 0 && !showHistory && (
        <div className="flex-shrink-0 px-4 sm:px-6 py-1.5 border-b border-border bg-muted/20">
          <p className="text-[10px] text-muted-foreground max-w-3xl mx-auto">
            <span className="font-medium">Data available:</span>{" "}
            {muniNames.join(", ")}
          </p>
        </div>
      )}

      {/* Messages -- native scrolling div instead of ScrollArea */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-3xl mx-auto space-y-4 px-4 sm:px-6 py-4">
          {messages.map((msg, i) => (
            <MessageBubble key={`${activeId}-${i}`} message={msg} />
          ))}

          {/* Suggested questions on first message */}
          {messages.length === 1 && !loading && (
            <div className="flex flex-wrap gap-1.5 pl-8 animate-fade-in-up" style={{ animationDelay: "200ms" }}>
              {SUGGESTED_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSend(q)}
                  className="text-[11px] bg-muted text-muted-foreground px-2.5 py-1 rounded-full border border-border hover:bg-accent hover:text-accent-foreground transition-colors text-left"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* Active thinking process */}
          {loading && (
            <div className="animate-slide-in-left">
              <div className="flex gap-2">
                <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="w-3.5 h-3.5 text-primary" />
                </div>
                <div className="flex-1 max-w-[88%]">
                  {currentThoughts.length > 0 && (
                    <div className="space-y-2 mb-2">
                      {currentThoughts.map((step, i) => (
                        <ThoughtStepLive key={i} step={step} index={i} />
                      ))}
                    </div>
                  )}

                  {streamingContent ? (
                    <div className="bg-muted/50 border border-border rounded-xl px-3 py-2 animate-fade-in">
                      <div className="prose prose-sm max-w-none prose-headings:text-sm prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1 prose-p:text-[13px] prose-p:leading-relaxed prose-ul:my-1 prose-li:my-0 prose-li:text-[13px] prose-strong:text-foreground prose-code:text-xs prose-td:text-xs prose-th:text-xs">
                        <ReactMarkdown>{streamingContent}</ReactMarkdown>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-muted/50 border border-border rounded-xl px-3 py-2">
                      <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
                        <div className="thinking-dots">
                          <span />
                          <span />
                          <span />
                        </div>
                        <span>
                          {currentThoughts.length === 0
                            ? "Analyzing your question..."
                            : "Thinking..."}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-border px-4 sm:px-6 py-3">
        <div className="max-w-3xl mx-auto flex gap-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about zoning..."
            className="text-sm h-10"
            disabled={loading}
          />
          <Button
            size="icon"
            className="h-10 w-10 shrink-0"
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ThoughtStepLive -- shown during streaming
// ---------------------------------------------------------------------------
function ThoughtStepLive({ step, index }: { step: ThoughtStep; index: number }) {
  const Icon = TOOL_ICONS[step.name] || Search;
  return (
    <div
      className="animate-slide-in-left"
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Header row */}
      <div className="flex items-start gap-2 text-[12px]">
        <div
          className={`w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${
            step.done
              ? "bg-green-100 text-green-600"
              : "bg-primary/10 text-primary animate-orb-pulse"
          }`}
        >
          <Icon className="w-2.5 h-2.5" />
        </div>
        <div className="flex-1 min-w-0">
          <span className={step.done ? "text-muted-foreground" : "text-foreground font-medium"}>
            {step.description}
          </span>
          {step.done && step.summary && (
            <span className="text-muted-foreground ml-1.5">
              &mdash; {step.summary}
            </span>
          )}
        </div>
      </div>

      {/* Detail: SQL query / search term */}
      {step.detail && (
        <div className="ml-6 mt-1 animate-fade-in">
          <div className="font-mono text-[10px] text-muted-foreground bg-muted/80 px-2 py-1.5 rounded-md border border-border/50 overflow-x-auto max-w-full whitespace-pre-wrap break-all leading-relaxed">
            {step.detail}
          </div>
        </div>
      )}

      {/* Preview: result summary */}
      {step.done && step.preview && (
        <div className="ml-6 mt-1 animate-fade-in">
          <div className="text-[10px] text-muted-foreground/70 whitespace-pre-wrap leading-relaxed pl-2 border-l-2 border-green-200">
            {step.preview}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SourceList — actionable sources with links and expandable SQL
// ---------------------------------------------------------------------------
function SourceList({ sources }: { sources: Source[] }) {
  const [expandedSql, setExpandedSql] = useState<number | null>(null);

  return (
    <div className="mt-2 pt-1.5 border-t border-border/50">
      <p className="text-[10px] text-muted-foreground mb-1.5 font-medium uppercase tracking-wider">
        Sources
      </p>
      <div className="space-y-1">
        {sources.slice(0, 8).map((s, j) => {
          const cfg = SOURCE_CONFIG[s.type || "text"] || DEFAULT_SOURCE;
          const Icon = cfg.icon;

          if (s.type === "sql") {
            const isExpanded = expandedSql === j;
            return (
              <div key={j}>
                <button
                  onClick={() => setExpandedSql(isExpanded ? null : j)}
                  className={`w-full flex items-center gap-1.5 text-[11px] border px-2 py-1 rounded-md transition-colors hover:bg-accent/50 text-left ${cfg.className}`}
                >
                  <Icon className="w-3 h-3 flex-shrink-0" />
                  <span className="flex-1 truncate">{s.source_document || "SQL Query"}</span>
                  <ChevronDown className={`w-3 h-3 opacity-50 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                </button>
                {isExpanded && s.sql && (
                  <pre className="mt-1 ml-4 text-[10px] bg-muted/80 border border-border rounded-md p-2 overflow-x-auto font-mono text-muted-foreground whitespace-pre-wrap">
                    {s.sql}
                  </pre>
                )}
              </div>
            );
          }

          if (s.source_url) {
            return (
              <a
                key={j}
                href={s.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center gap-1.5 text-[11px] border px-2 py-1 rounded-md transition-colors hover:bg-accent/50 ${cfg.className}`}
              >
                <Icon className="w-3 h-3 flex-shrink-0" />
                <span className="flex-1 truncate">{s.source_document || s.source_url}</span>
                <ExternalLink className="w-3 h-3 opacity-40 flex-shrink-0" />
              </a>
            );
          }

          return (
            <span
              key={j}
              className={`flex items-center gap-1.5 text-[11px] border px-2 py-1 rounded-md ${cfg.className}`}
            >
              <Icon className="w-3 h-3 flex-shrink-0" />
              <span className="flex-1 truncate">
                {s.municipality
                  ? `${s.municipality}${s.zone_code ? ` / ${s.zone_code}` : ""}`
                  : s.source_document || cfg.label}
              </span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MessageBubble
// ---------------------------------------------------------------------------
function MessageBubble({ message: msg }: { message: Message }) {
  const [thoughtsOpen, setThoughtsOpen] = useState(false);

  return (
    <div
      className={`flex gap-2 ${msg.role === "user" ? "justify-end animate-slide-in-right" : "animate-slide-in-left"}`}
    >
      {msg.role === "assistant" && (
        <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Bot className="w-3.5 h-3.5 text-primary" />
        </div>
      )}
      <div
        className={`max-w-[88%] rounded-xl px-3 py-2 ${
          msg.role === "user"
            ? "bg-primary text-primary-foreground"
            : "bg-muted/50 border border-border"
        }`}
      >
        {msg.thoughts && msg.thoughts.length > 0 && (
          <button
            onClick={() => setThoughtsOpen((prev) => !prev)}
            className="flex items-center gap-1 text-[10px] text-muted-foreground mb-1.5 hover:text-foreground transition-colors"
          >
            <ChevronDown
              className={`w-3 h-3 transition-transform ${thoughtsOpen ? "rotate-0" : "-rotate-90"}`}
            />
            {thoughtsOpen ? "Hide" : "Show"} reasoning ({msg.thoughts.length} steps)
          </button>
        )}
        {msg.thoughts && thoughtsOpen && (
          <div className="space-y-2 mb-2 pb-2 border-b border-border/50 animate-fade-in">
            {msg.thoughts.map((step, i) => {
              const Icon = TOOL_ICONS[step.name] || Search;
              return (
                <div key={i} className="text-[10px] text-muted-foreground">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3.5 h-3.5 rounded-full bg-green-100 text-green-600 flex items-center justify-center flex-shrink-0">
                      <Icon className="w-2 h-2" />
                    </div>
                    <span className="font-medium">{step.description}</span>
                    {step.summary && (
                      <span className="opacity-70">&mdash; {step.summary}</span>
                    )}
                  </div>
                  {step.detail && (
                    <div className="ml-5 mt-0.5 font-mono text-[9px] bg-muted/80 px-2 py-1 rounded border border-border/50 overflow-x-auto max-w-full whitespace-pre-wrap break-all">
                      {step.detail}
                    </div>
                  )}
                  {step.preview && (
                    <div className="ml-5 mt-0.5 text-[9px] opacity-60 whitespace-pre-wrap">
                      {step.preview}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {msg.role === "assistant" ? (
          <div className="prose prose-sm max-w-none prose-headings:text-sm prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1 prose-p:text-[13px] prose-p:leading-relaxed prose-ul:my-1 prose-li:my-0 prose-li:text-[13px] prose-strong:text-foreground prose-code:text-xs prose-td:text-xs prose-th:text-xs">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-[13px]">{msg.content}</p>
        )}

        {msg.sources && msg.sources.length > 0 && (
          <SourceList sources={msg.sources} />
        )}
      </div>
      {msg.role === "user" && (
        <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center flex-shrink-0 mt-0.5">
          <User className="w-3.5 h-3.5 text-muted-foreground" />
        </div>
      )}
    </div>
  );
}
