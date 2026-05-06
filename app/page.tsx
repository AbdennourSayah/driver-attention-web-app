import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Eye,
  Brain,
  Database,
  ArrowRight,
  Sparkles,
  Cpu,
  Activity,
  Layers,
} from "lucide-react";

const heroExamples = [
  {
    src: "/images/saliency-example-1.png",
    label: "Highway",
    alt: "Highway saliency overlay",
  },
  {
    src: "/images/saliency-example-2.jpg",
    label: "Urban",
    alt: "Urban saliency overlay",
  },
  {
    src: "/images/saliency-example-3.jpg",
    label: "Crowded",
    alt: "Crowded saliency overlay",
  },
];

const systemFeatures = [
  {
    icon: Eye,
    title: "Saliency Prediction",
    body:
      "Per-frame heatmaps highlighting where a driver is most likely to look — vehicles, signs, and road hazards.",
  },
  {
    icon: Brain,
    title: "Spatio-Temporal Modeling",
    body:
      "An R3D-18 backbone consumes 16-frame clips so the network learns motion cues, not just static appearance.",
  },
  {
    icon: Layers,
    title: "Encoder–Decoder",
    body:
      "Four upsampling stages with skip connections from temporally pooled encoder features and dropout (p=0.30).",
  },
];

const stats = [
  { value: "74", label: "Video Sequences" },
  { value: "500K+", label: "Frames" },
  { value: "8", label: "Drivers" },
  { value: "16", label: "Frames / Clip" },
];

export default function HomePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-12 md:pt-20">
      {/* Hero */}
      <section className="mb-24">
        <div className="grid items-center gap-10 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <Badge variant="outline" className="mb-5 gap-1.5 border-primary/30 bg-primary/5 text-primary">
              <Sparkles className="h-3 w-3" />
              Master Thesis Research · 2024
            </Badge>

            <h1 className="mb-5 text-balance text-4xl font-bold leading-tight tracking-tight md:text-6xl">
              Where do drivers{" "}
              <span className="bg-gradient-to-r from-primary via-primary to-accent bg-clip-text text-transparent">
                look
              </span>
              ?
            </h1>

            <p className="mb-8 max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground">
              A spatio-temporal CNN trained on the DR(eye)VE dataset that
              predicts per-frame driver-attention saliency maps from raw
              dashcam footage. Upload an image or a clip — see the model&apos;s
              attention in seconds.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="group">
                <Link href="/predict">
                  Try the model
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="/about">How it works</Link>
              </Button>
              <a
                href="https://aimagelab.ing.unimore.it/dreyeve/"
                target="_blank"
                rel="noreferrer"
                className="ml-1 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              >
                DR(eye)VE dataset →
              </a>
            </div>
          </div>

          {/* Stacked example cards */}
          <div className="relative aspect-[4/3] w-full">
            <div className="absolute inset-0 -translate-y-2 translate-x-3 rotate-2 rounded-2xl border border-border bg-card/40 shadow-lg" />
            <div className="absolute inset-0 translate-y-3 -translate-x-3 -rotate-1 rounded-2xl border border-border bg-card/40 shadow-lg" />
            <div className="absolute inset-0 overflow-hidden rounded-2xl border border-border bg-card shadow-2xl">
              <Image
                src="/images/saliency-example-1.png"
                alt="Driver attention saliency overlay"
                width={900}
                height={600}
                className="h-full w-full object-cover"
                priority
              />
              <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/70 via-black/30 to-transparent px-4 py-3">
                <div className="flex items-center gap-2 text-xs font-medium text-white">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                  </span>
                  Predicted saliency
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-white/70">
                  <Cpu className="h-3 w-3" />
                  R3D-18 · 192×112 · 16 frames
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats strip */}
      <section className="mb-24">
        <Card className="border-border/60 bg-card/60 backdrop-blur-sm">
          <CardContent className="grid grid-cols-2 gap-6 p-6 sm:grid-cols-4">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center sm:text-left">
                <div className="text-3xl font-bold tracking-tight text-primary">
                  {stat.value}
                </div>
                <div className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">
                  {stat.label}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      {/* From scene to attention */}
      <section className="mb-24">
        <div className="mb-8 flex flex-col items-start gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
              From driving scene to{" "}
              <span className="text-primary">attention map</span>
            </h2>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              The same checkpoint serves images and videos. Below: three
              representative samples produced by the RGB baseline.
            </p>
          </div>
          <Button asChild variant="ghost" size="sm" className="group">
            <Link href="/predict">
              Run on your own image
              <ArrowRight className="ml-1.5 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          {heroExamples.map((ex) => (
            <Card
              key={ex.src}
              className="group overflow-hidden border-border/60 bg-card/60 transition-all hover:border-primary/40 hover:shadow-lg"
            >
              <div className="relative aspect-video overflow-hidden">
                <Image
                  src={ex.src}
                  alt={ex.alt}
                  fill
                  sizes="(min-width: 768px) 33vw, 100vw"
                  className="object-cover transition-transform duration-500 group-hover:scale-105"
                />
                <div className="absolute left-3 top-3 rounded-full bg-black/60 px-2.5 py-0.5 text-[11px] font-medium text-white backdrop-blur">
                  {ex.label}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* What it does */}
      <section className="mb-24">
        <h2 className="mb-8 text-2xl font-semibold tracking-tight md:text-3xl">
          What the model <span className="text-primary">does</span>
        </h2>

        <div className="grid gap-4 md:grid-cols-3">
          {systemFeatures.map(({ icon: Icon, title, body }) => (
            <Card
              key={title}
              className="border-border/60 bg-card/60 transition-colors hover:border-primary/40"
            >
              <CardHeader>
                <div className="mb-2 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Icon className="h-5 w-5" />
                </div>
                <CardTitle className="text-lg">{title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                {body}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Dataset card */}
      <section className="mb-24">
        <Card className="overflow-hidden border-border/60 bg-card/60">
          <div className="grid lg:grid-cols-2">
            <div className="p-8">
              <div className="mb-4 flex items-center gap-3">
                <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Database className="h-5 w-5" />
                </div>
                <div>
                  <CardTitle className="text-xl">DR(eye)VE Dataset</CardTitle>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    Training foundation
                  </p>
                </div>
              </div>

              <p className="mb-6 leading-relaxed text-muted-foreground">
                Built on the DR(eye)VE benchmark — synchronized dashcam video
                and eye-tracking from 8 drivers across diverse road conditions.
                The model learns from these ground-truth fixation maps to
                generalize attention prediction to unseen drives.
              </p>

              <div className="grid grid-cols-3 gap-3">
                {stats.slice(0, 3).map((stat) => (
                  <div
                    key={stat.label}
                    className="rounded-lg border border-border/60 bg-secondary/40 p-3 text-center"
                  >
                    <div className="text-xl font-bold text-primary">
                      {stat.value}
                    </div>
                    <div className="mt-0.5 text-[11px] text-muted-foreground">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="relative min-h-[280px] overflow-hidden bg-secondary/40">
              <Image
                src="/images/saliency-example-2.jpg"
                alt="Dataset preview"
                fill
                sizes="(min-width: 1024px) 50vw, 100vw"
                className="object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-l from-transparent via-card/0 to-card/40 lg:to-card/80" />
            </div>
          </div>
        </Card>
      </section>

      {/* Pipeline cards */}
      <section>
        <h2 className="mb-8 text-2xl font-semibold tracking-tight md:text-3xl">
          Explore the <span className="text-primary">system</span>
        </h2>

        <div className="grid gap-4 md:grid-cols-3">
          <Link href="/predict" className="group block">
            <Card className="h-full border-border/60 bg-card/60 transition-all hover:-translate-y-1 hover:border-primary/40 hover:shadow-xl">
              <CardHeader>
                <Activity className="mb-3 h-7 w-7 text-primary" />
                <CardTitle className="flex items-center justify-between text-lg">
                  Saliency Prediction
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Upload a driving image or clip and get the predicted attention
                map back as PNG / MP4.
              </CardContent>
            </Card>
          </Link>

          <Link href="/chat" className="group block">
            <Card className="h-full border-border/60 bg-card/60 transition-all hover:-translate-y-1 hover:border-primary/40 hover:shadow-xl">
              <CardHeader>
                <Brain className="mb-3 h-7 w-7 text-primary" />
                <CardTitle className="flex items-center justify-between text-lg">
                  Thesis Assistant
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Ask about the architecture, evaluation metrics, or the
                DR(eye)VE dataset.
              </CardContent>
            </Card>
          </Link>

          <Link href="/about" className="group block">
            <Card className="h-full border-border/60 bg-card/60 transition-all hover:-translate-y-1 hover:border-primary/40 hover:shadow-xl">
              <CardHeader>
                <Eye className="mb-3 h-7 w-7 text-primary" />
                <CardTitle className="flex items-center justify-between text-lg">
                  About the Model
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Architecture, training loss, normalization, and known
                limitations.
              </CardContent>
            </Card>
          </Link>
        </div>
      </section>
    </div>
  );
}
