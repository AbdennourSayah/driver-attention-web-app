import Link from "next/link";
import { Github, BookOpen } from "lucide-react";

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-border bg-background/60">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10 text-sm md:flex-row md:items-center md:justify-between">
        <div>
          <p className="font-semibold tracking-tight">
            dr(<span className="text-primary">eye</span>)ve
          </p>
          <p className="mt-1 max-w-md text-muted-foreground">
            Master&apos;s thesis prototype on driver attention prediction with
            spatio-temporal CNNs. Research only — never use as a driving aid.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-muted-foreground">
          <Link
            href="/predict"
            className="hover:text-foreground transition-colors"
          >
            Prediction
          </Link>
          <Link
            href="/chat"
            className="hover:text-foreground transition-colors"
          >
            Chatbot
          </Link>
          <Link
            href="/about"
            className="hover:text-foreground transition-colors"
          >
            About
          </Link>
          <a
            href="https://aimagelab.ing.unimore.it/dreyeve/"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
          >
            <BookOpen className="h-3.5 w-3.5" />
            DR(eye)VE
          </a>
          <a
            href="https://github.com/AbdennourSayah/driver-attention-web-app"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
          >
            <Github className="h-3.5 w-3.5" />
            Source
          </a>
        </div>
      </div>
    </footer>
  );
}
