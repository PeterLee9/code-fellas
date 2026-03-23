"use client";

import { useState, useCallback, useEffect, createContext, useContext } from "react";
import { Navbar } from "@/components/Navbar";
import { ChatPanel } from "@/components/ChatPanel";
import { Sparkles, Command } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatContextValue {
  openChat: () => void;
  closeChat: () => void;
  sendQuestion: (question: string) => void;
  isChatOpen: boolean;
}

const ChatContext = createContext<ChatContextValue>({
  openChat: () => {},
  closeChat: () => {},
  sendQuestion: () => {},
  isChatOpen: false,
});

export function useChatContext() {
  return useContext(ChatContext);
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [chatOpen, setChatOpen] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | undefined>();

  const openChat = useCallback(() => setChatOpen(true), []);
  const closeChat = useCallback(() => setChatOpen(false), []);

  const sendQuestion = useCallback((question: string) => {
    setPendingQuestion(question);
    setChatOpen(true);
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setChatOpen(true);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <ChatContext.Provider value={{ openChat, closeChat, sendQuestion, isChatOpen: chatOpen }}>
      <div className="flex flex-col h-full bg-background text-foreground">
        <Navbar />

        <div className="flex-1 overflow-hidden">
          {chatOpen ? (
            <ChatPanel
              isOpen
              onClose={closeChat}
              initialQuestion={pendingQuestion}
              onQuestionConsumed={() => setPendingQuestion(undefined)}
            />
          ) : (
            <main className="h-full overflow-y-auto">{children}</main>
          )}
        </div>

        {/* Floating "Ask AI" button -- only when chat is closed */}
        {!chatOpen && (
          <Button
            onClick={openChat}
            className="fixed bottom-5 right-5 z-40 h-12 gap-2 rounded-full px-5 shadow-lg hover:shadow-xl transition-all animate-scale-in lg:bottom-6 lg:right-6"
            size="lg"
          >
            <Sparkles className="w-4 h-4" />
            <span className="hidden sm:inline">Ask AI</span>
            <kbd className="hidden md:inline-flex items-center gap-0.5 ml-1 text-[10px] opacity-60 bg-primary-foreground/10 px-1.5 py-0.5 rounded font-mono">
              <Command className="w-2.5 h-2.5" />K
            </kbd>
          </Button>
        )}
      </div>
    </ChatContext.Provider>
  );
}
