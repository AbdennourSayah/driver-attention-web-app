"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { sendChatMessage, type ChatMessage } from "@/lib/api";
import { Send, Bot, User, Loader2, Lightbulb } from "lucide-react";
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
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

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
      await new Promise((resolve) => setTimeout(resolve, 800));
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: getMockResponse(text) },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold tracking-tight">
          Thesis Assistant
        </h1>
        <p className="text-muted-foreground">
          Ask questions about driver attention prediction, saliency maps, and
          the research methodology.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Chat Area */}
        <Card className="bg-card lg:col-span-2">
          <CardContent className="flex h-[500px] flex-col p-0">
            <ScrollArea className="flex-1 p-4" ref={scrollRef}>
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
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Bot className="h-4 w-4" />
                      </div>
                    )}
                    <div
                      className={cn(
                        "max-w-[80%] rounded-lg px-4 py-2 text-sm",
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-secondary text-secondary-foreground"
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
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="flex items-center rounded-lg bg-secondary px-4 py-2">
                      <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>

            <div className="border-t border-border p-4">
              <div className="flex gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about driver attention, saliency maps..."
                  disabled={isLoading}
                  className="flex-1"
                />
                <Button
                  onClick={() => handleSend()}
                  disabled={!input.trim() || isLoading}
                  size="icon"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Suggested Questions */}
        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Lightbulb className="h-4 w-4 text-primary" />
              Suggested Questions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-2">
              {suggestedQuestions.map((question, index) => (
                <Button
                  key={index}
                  variant="ghost"
                  className="h-auto justify-start whitespace-normal px-3 py-2 text-left text-sm font-normal text-muted-foreground hover:text-foreground"
                  onClick={() => handleSend(question)}
                  disabled={isLoading}
                >
                  {question}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
