"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { sendChatMessage, type ChatMessage } from "@/lib/api";
import { Send, Bot, User, Loader2, Lightbulb, MessageCircle, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const suggestedQuestions = [
  "What is driver attention prediction?",
  "Explain saliency maps in this context",
  "What is the DR(eye)VE dataset?",
  "How does the RGB model work?",
  "What are CC, KL, and IG metrics?",
  "What is optical flow?",
  "Explain the 16-frame input approach",
  "What are the model limitations?",
];

// Mock responses for demo when backend is not connected
const mockResponses: Record<string, string> = {
  default:
    "I'm the thesis assistant for the driver attention prediction research. I can help answer questions about saliency maps, the DR(eye)VE dataset, model architecture, evaluation metrics, and more. What would you like to know?",
  attention:
    "Driver attention prediction is the task of estimating where a driver is likely to look while driving. This research uses deep learning to predict visual saliency maps from driving footage, highlighting areas that typically attract driver attention such as other vehicles, pedestrians, traffic signs, and road hazards.",
  saliency:
    "Saliency maps are 2D heatmaps that represent visual attention. In this research, they show the probability distribution of where a driver's gaze is likely to be focused. Brighter areas indicate higher attention probability. The model generates these maps by learning from real eye-tracking data collected from drivers.",
  dreyeve:
    "The DR(eye)VE dataset is a comprehensive benchmark for driver attention research. It contains 74 video sequences (over 500,000 frames) recorded from 8 different drivers in various road conditions. Each frame has corresponding eye-tracking data providing ground truth attention maps for training and evaluation.",
  rgb: "The RGB model uses only the visual appearance of driving scenes as input. It employs a CNN encoder-decoder architecture that processes RGB frames and outputs predicted saliency maps. While simpler than multi-modal approaches, it captures essential visual cues like object appearance, color, and spatial layout.",
  metrics:
    "The model is evaluated using several metrics: CC (Correlation Coefficient) measures linear correlation between predicted and ground truth maps. KL (Kullback-Leibler divergence) measures how one probability distribution diverges from another. IG (Information Gain) measures the information gained from the predicted map over a baseline.",
  flow: "Optical flow captures motion information between consecutive frames. It represents pixel-wise displacement vectors showing how objects move. In driver attention, motion cues are important because moving objects (other vehicles, pedestrians) strongly attract attention.",
  frames:
    "The model processes 16-frame temporal clips to capture attention dynamics over time. This approach allows the network to learn how attention evolves during driving, capturing patterns like gaze shifts when approaching intersections or tracking moving objects.",
  limitations:
    "Current limitations include: (1) This is a research prototype, not a safety-critical system. (2) Performance varies with weather and lighting conditions. (3) The model may not generalize well to driving scenarios very different from the training data. (4) Real-time processing may require GPU acceleration.",
};

function getMockResponse(message: string): string {
  const lower = message.toLowerCase();
  if (lower.includes("attention") && lower.includes("prediction"))
    return mockResponses.attention;
  if (lower.includes("saliency")) return mockResponses.saliency;
  if (lower.includes("dreyeve") || lower.includes("dr(eye)ve") || lower.includes("dataset"))
    return mockResponses.dreyeve;
  if (lower.includes("rgb") && lower.includes("model")) return mockResponses.rgb;
  if (lower.includes("metric") || lower.includes("cc") || lower.includes("kl") || lower.includes("ig"))
    return mockResponses.metrics;
  if (lower.includes("optical") || lower.includes("flow"))
    return mockResponses.flow;
  if (lower.includes("16") || lower.includes("frame") || lower.includes("temporal"))
    return mockResponses.frames;
  if (lower.includes("limitation") || lower.includes("restrict"))
    return mockResponses.limitations;
  return mockResponses.default;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm the thesis assistant for the driver attention prediction research. I can answer questions about saliency maps, the DR(eye)VE dataset, model architecture, evaluation metrics (CC, KL, IG), and more. How can I help you?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  const handleSend = async (message?: string) => {
    const text = message || input.trim();
    if (!text) return;

    const userMessage: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendChatMessage(text, messages);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.response },
      ]);
    } catch {
      // Use mock response when backend is not available
      await new Promise((resolve) => setTimeout(resolve, 600));
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: getMockResponse(text) },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([
      {
        role: "assistant",
        content:
          "Hi again! Ask me anything about the model, training, or the DR(eye)VE dataset.",
      },
    ]);
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-12">
      <div className="mb-8 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <Badge variant="outline" className="mb-3 gap-1.5 border-primary/30 bg-primary/5 text-primary">
            <MessageCircle className="h-3 w-3" />
            Q&amp;A assistant
          </Badge>
          <h1 className="mb-2 text-3xl font-bold tracking-tight md:text-4xl">
            Thesis Assistant
          </h1>
          <p className="max-w-xl text-muted-foreground">
            Ask anything about driver attention prediction, the architecture,
            evaluation metrics, or the DR(eye)VE dataset.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={handleClear}>
          New chat
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Chat Area */}
        <Card className="bg-card/60 backdrop-blur-sm lg:col-span-2">
          <CardContent className="flex h-[560px] flex-col p-0">
            <ScrollArea className="flex-1 p-4">
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={cn(
                      "flex gap-3",
                      message.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    {message.role === "assistant" && (
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm">
                        <Bot className="h-4 w-4" />
                      </div>
                    )}
                    <div
                      className={cn(
                        "max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm",
                        message.role === "user"
                          ? "rounded-br-sm bg-primary text-primary-foreground"
                          : "rounded-bl-sm border border-border bg-secondary text-secondary-foreground"
                      )}
                    >
                      {message.content}
                    </div>
                    {message.role === "user" && (
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                        <User className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                ))}
                {isLoading && (
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="flex items-center rounded-2xl rounded-bl-sm border border-border bg-secondary px-4 py-2.5 text-sm">
                      <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    </div>
                  </div>
                )}
                <div ref={bottomRef} aria-hidden />
              </div>
            </ScrollArea>

            <div className="border-t border-border p-3">
              <div className="flex gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about driver attention, saliency maps…"
                  disabled={isLoading}
                  className="flex-1"
                />
                <Button
                  onClick={() => handleSend()}
                  disabled={!input.trim() || isLoading}
                  size="icon"
                  aria-label="Send message"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground">
                Falls back to canned answers when the backend{" "}
                <code className="font-mono">/chat</code> endpoint is not
                reachable.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Suggested Questions */}
        <Card className="bg-card/60 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Lightbulb className="h-4 w-4 text-primary" />
              Suggested questions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-2">
              {suggestedQuestions.map((question) => (
                <Button
                  key={question}
                  variant="ghost"
                  className="h-auto justify-start whitespace-normal rounded-lg px-3 py-2 text-left text-sm font-normal text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                  onClick={() => handleSend(question)}
                  disabled={isLoading}
                >
                  <Sparkles className="mr-2 h-3.5 w-3.5 shrink-0 text-primary/70" />
                  <span className="flex-1">{question}</span>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
