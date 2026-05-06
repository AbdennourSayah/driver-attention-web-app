import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Eye, Brain, Video, Database, ArrowRight } from "lucide-react";

export default function HomePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16">
      {/* Hero Section */}
      <section className="mb-20">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div>
            <div className="mb-6 flex items-center gap-2">
              <span className="text-sm font-medium text-muted-foreground">
                Master Thesis Research
              </span>
              <span className="h-1 w-1 rounded-full bg-primary" />
              <span className="text-sm font-medium text-primary">2024</span>
            </div>

            <h1 className="mb-6 text-balance text-4xl font-bold leading-tight tracking-tight md:text-5xl">
              Analysis and Prediction of{" "}
              <span className="text-primary">Driver Attention</span> in Real Driving Scenarios
            </h1>

            <p className="mb-8 text-pretty text-lg leading-relaxed text-muted-foreground">
              A deep learning system that predicts where drivers look while driving.
              Using convolutional neural networks and temporal modeling, this
              research analyzes visual attention patterns to generate accurate
              saliency maps from driving footage.
            </p>

            <div className="flex flex-wrap gap-4">
              <Button asChild size="lg">
                <Link href="/predict">
                  Try the Model
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="/about">Learn More</Link>
              </Button>
            </div>
          </div>

          <div className="relative">
            <div className="overflow-hidden rounded-xl border border-border bg-card p-2">
              <Image
                src="/images/saliency-overlay.jpg"
                alt="Driver attention saliency map overlay on driving scene"
                width={600}
                height={400}
                className="rounded-lg"
              />
            </div>
            <div className="absolute -bottom-4 -left-4 z-10 overflow-hidden rounded-lg border border-border bg-card p-1.5 shadow-xl">
              <Image
                src="/images/saliency-map-1.jpg"
                alt="Saliency heatmap visualization"
                width={180}
                height={120}
                className="rounded"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Visual Examples Section */}
      <section className="mb-20">
        <h2 className="mb-8 text-2xl font-semibold tracking-tight">
          From Driving Scene to <span className="text-primary">Attention Map</span>
        </h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          <Card className="overflow-hidden bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium">Input: Driving Scene</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Image
                src="/images/driving-scene-1.jpg"
                alt="Highway driving scene from driver perspective"
                width={600}
                height={340}
                className="w-full"
              />
            </CardContent>
          </Card>

          <Card className="overflow-hidden bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium">Output: Predicted Saliency</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Image
                src="/images/saliency-map-1.jpg"
                alt="Predicted attention saliency map"
                width={600}
                height={340}
                className="w-full"
              />
            </CardContent>
          </Card>
        </div>
      </section>

      {/* What is Driver Attention Prediction */}
      <section className="mb-20">
        <h2 className="mb-8 text-2xl font-semibold tracking-tight">
          Understanding Driver <span className="text-primary">Visual Attention</span>
        </h2>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <Card className="bg-card">
            <CardHeader>
              <Eye className="mb-2 h-8 w-8 text-primary" />
              <CardTitle className="text-lg">Saliency Prediction</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              The model generates attention heatmaps showing where a driver is
              likely to focus their gaze. These saliency maps highlight critical
              road elements like vehicles, pedestrians, and traffic signs.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <Brain className="mb-2 h-8 w-8 text-primary" />
              <CardTitle className="text-lg">Deep Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Using CNN encoder-decoder architecture with temporal modeling
              across 16-frame clips, the system learns attention patterns from
              real driving data collected with eye-tracking devices.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <Video className="mb-2 h-8 w-8 text-primary" />
              <CardTitle className="text-lg">Multi-Modal Input</CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              The architecture supports RGB frames, optical flow for motion
              cues, and semantic segmentation masks to capture comprehensive
              scene understanding for attention prediction.
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Dataset Section */}
      <section className="mb-20">
        <Card className="bg-card">
          <CardHeader>
            <div className="flex items-center gap-3">
              <Database className="h-8 w-8 text-primary" />
              <div>
                <CardTitle className="text-xl">DR(eye)VE Dataset</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Training Foundation
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <p className="mb-6 leading-relaxed text-muted-foreground">
                  This research is built on the DR(eye)VE dataset, a comprehensive
                  collection of driving sequences with synchronized eye-tracking
                  data. The dataset provides ground truth attention maps captured
                  from real drivers in various road conditions.
                </p>

                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-lg bg-secondary/50 p-4">
                    <div className="text-2xl font-bold text-primary">74</div>
                    <div className="text-sm text-muted-foreground">
                      Video Sequences
                    </div>
                  </div>
                  <div className="rounded-lg bg-secondary/50 p-4">
                    <div className="text-2xl font-bold text-primary">500K+</div>
                    <div className="text-sm text-muted-foreground">Frames</div>
                  </div>
                  <div className="rounded-lg bg-secondary/50 p-4">
                    <div className="text-2xl font-bold text-primary">8</div>
                    <div className="text-sm text-muted-foreground">Drivers</div>
                  </div>
                </div>
              </div>

              <div className="overflow-hidden rounded-lg">
                <Image
                  src="/images/driving-scene-2.jpg"
                  alt="Urban driving scene example from dataset"
                  width={500}
                  height={280}
                  className="h-full w-full object-cover"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Features Grid */}
      <section>
        <h2 className="mb-8 text-2xl font-semibold tracking-tight">
          Explore the <span className="text-primary">System</span>
        </h2>

        <div className="grid gap-4 md:grid-cols-3">
          <Link href="/predict" className="group">
            <Card className="h-full bg-card transition-colors hover:border-primary/50">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Saliency Prediction
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </CardTitle>
              </CardHeader>
              <CardContent className="text-muted-foreground">
                Upload driving images or videos and see predicted attention maps.
              </CardContent>
            </Card>
          </Link>

          <Link href="/chat" className="group">
            <Card className="h-full bg-card transition-colors hover:border-primary/50">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Thesis Assistant
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </CardTitle>
              </CardHeader>
              <CardContent className="text-muted-foreground">
                Ask questions about driver attention and the model architecture.
              </CardContent>
            </Card>
          </Link>

          <Link href="/about" className="group">
            <Card className="h-full bg-card transition-colors hover:border-primary/50">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  About the Model
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </CardTitle>
              </CardHeader>
              <CardContent className="text-muted-foreground">
                Learn about the neural network architecture and limitations.
              </CardContent>
            </Card>
          </Link>
        </div>
      </section>
    </div>
  );
}
