import Image from "next/image";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Layers,
  Cpu,
  Timer,
  ImageIcon,
  AlertTriangle,
  FlaskConical,
  Gauge,
  GitBranch,
} from "lucide-react";

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      <div className="mb-12 grid items-center gap-8 lg:grid-cols-2">
        <div>
          <h1 className="mb-2 text-3xl font-bold tracking-tight">
            About the Model
          </h1>
          <p className="text-muted-foreground">
            Technical details about the driver attention prediction architecture
            and methodology.
          </p>
        </div>
        <div className="overflow-hidden rounded-xl border border-border">
          <Image
            src="/images/saliency-overlay.jpg"
            alt="Example of saliency prediction overlay"
            width={500}
            height={280}
            className="w-full"
          />
        </div>
      </div>

      {/* Architecture Overview */}
      <section className="mb-12">
        <h2 className="mb-6 text-xl font-semibold">Model Architecture</h2>

        <div className="grid gap-4 sm:grid-cols-2">
          <Card className="bg-card">
            <CardHeader>
              <ImageIcon className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">RGB Input</CardTitle>
              <CardDescription>Visual appearance processing</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              The model takes RGB frames from dashboard cameras as primary
              input. Images are resized to a fixed resolution and normalized
              before processing through the encoder network.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <Timer className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">Temporal Clips</CardTitle>
              <CardDescription>16-frame sequences</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              To capture temporal dynamics, the model processes 16 consecutive
              frames at a time. This allows learning attention patterns that
              evolve over time, such as tracking moving objects.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <Layers className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">CNN Encoder-Decoder</CardTitle>
              <CardDescription>Feature extraction and generation</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              The architecture uses a convolutional encoder to extract
              hierarchical features, followed by a decoder that generates
              pixel-wise saliency predictions. Skip connections preserve spatial
              details.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <Cpu className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">Saliency Output</CardTitle>
              <CardDescription>Attention heatmaps</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              The final output is a probability map where each pixel represents
              the likelihood of driver attention. Values are normalized to form
              a valid probability distribution.
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Multi-Modal Support */}
      <section className="mb-12">
        <h2 className="mb-6 text-xl font-semibold">Multi-Modal Inputs</h2>
        <Card className="bg-card">
          <CardContent className="pt-6">
            <div className="grid gap-6 sm:grid-cols-3">
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <Badge variant="default">Available</Badge>
                  <span className="font-medium">RGB</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Visual appearance features capturing color, texture, and
                  object shapes from the driving scene.
                </p>
              </div>
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <Badge variant="secondary">Planned</Badge>
                  <span className="font-medium">Optical Flow</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Motion information between frames, helping identify moving
                  objects that attract attention.
                </p>
              </div>
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <Badge variant="secondary">Planned</Badge>
                  <span className="font-medium">Semantic Segmentation</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Object category masks providing high-level scene understanding
                  for attention prediction.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Evaluation Metrics */}
      <section className="mb-12">
        <h2 className="mb-6 text-xl font-semibold">Evaluation Metrics</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          <Card className="bg-card">
            <CardHeader>
              <Gauge className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">CC</CardTitle>
              <CardDescription>Correlation Coefficient</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Measures linear correlation between predicted and ground truth
              saliency maps. Values range from -1 to 1, with higher being
              better.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <GitBranch className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">KL</CardTitle>
              <CardDescription>Kullback-Leibler Divergence</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Measures how the predicted probability distribution diverges from
              ground truth. Lower values indicate better alignment.
            </CardContent>
          </Card>

          <Card className="bg-card">
            <CardHeader>
              <FlaskConical className="mb-2 h-6 w-6 text-primary" />
              <CardTitle className="text-base">IG</CardTitle>
              <CardDescription>Information Gain</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Measures information gained from the predicted saliency map over a
              center-biased baseline. Higher values are better.
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Limitations */}
      <section className="mb-12">
        <h2 className="mb-6 text-xl font-semibold">Limitations</h2>

        <Alert variant="destructive" className="mb-4">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Research Prototype</AlertTitle>
          <AlertDescription>
            This system is a research prototype developed for academic purposes.
            It is NOT intended for use in real-world safety-critical
            applications or autonomous driving systems.
          </AlertDescription>
        </Alert>

        <Card className="bg-card">
          <CardContent className="pt-6">
            <ul className="space-y-4 text-sm text-muted-foreground">
              <li className="flex gap-3">
                <span className="font-medium text-foreground">1.</span>
                <span>
                  <strong className="text-foreground">
                    Generalization limits:
                  </strong>{" "}
                  The model is trained on the DR(eye)VE dataset which was
                  collected in specific geographic locations and weather
                  conditions. Performance may degrade on significantly different
                  driving scenarios.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="font-medium text-foreground">2.</span>
                <span>
                  <strong className="text-foreground">
                    Individual differences:
                  </strong>{" "}
                  The model predicts average attention patterns learned from
                  multiple drivers. Individual drivers may have different
                  attention behaviors based on experience and cognitive factors.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="font-medium text-foreground">3.</span>
                <span>
                  <strong className="text-foreground">
                    Environmental factors:
                  </strong>{" "}
                  Performance may vary with lighting conditions (night driving,
                  glare), weather (rain, fog, snow), and unusual road
                  configurations.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="font-medium text-foreground">4.</span>
                <span>
                  <strong className="text-foreground">
                    Computational requirements:
                  </strong>{" "}
                  Real-time processing requires GPU acceleration. Inference
                  speed depends on hardware capabilities and input resolution.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="font-medium text-foreground">5.</span>
                <span>
                  <strong className="text-foreground">Rare events:</strong> The
                  model may not accurately predict attention for rare or unusual
                  driving situations that were underrepresented in training
                  data.
                </span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </section>

      {/* Technical Stack */}
      <section>
        <h2 className="mb-6 text-xl font-semibold">Technical Stack</h2>
        <Card className="bg-card">
          <CardContent className="pt-6">
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">PyTorch</Badge>
              <Badge variant="outline">FastAPI</Badge>
              <Badge variant="outline">Next.js</Badge>
              <Badge variant="outline">Python</Badge>
              <Badge variant="outline">CUDA</Badge>
              <Badge variant="outline">OpenCV</Badge>
              <Badge variant="outline">NumPy</Badge>
              <Badge variant="outline">Tailwind CSS</Badge>
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The backend uses FastAPI with PyTorch for model inference. The
              frontend is built with Next.js and connects to the prediction API
              endpoints for image and video processing.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
