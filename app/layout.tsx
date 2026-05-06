import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";
import { Navigation } from "@/components/navigation";
import { SiteFooter } from "@/components/site-footer";
import { Toaster } from "@/components/ui/sonner";

const _geist = Geist({ subsets: ["latin"] });
const _geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "dr(eye)ve — Driver Attention Prediction",
  description:
    "Analysis and prediction of driver attention in real driving scenarios using a spatio-temporal R3D-18 saliency network.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased min-h-screen bg-background flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {/* Subtle radial gradient ambience tinted by the primary color */}
          <div
            aria-hidden
            className="pointer-events-none fixed inset-0 -z-10 opacity-60 [background:radial-gradient(60%_50%_at_50%_-10%,color-mix(in_oklch,var(--primary)_24%,transparent),transparent_70%),radial-gradient(40%_30%_at_100%_100%,color-mix(in_oklch,var(--primary)_18%,transparent),transparent_70%)]"
          />
          <Navigation />
          <main className="flex-1">{children}</main>
          <SiteFooter />
          <Toaster richColors position="top-right" />
        </ThemeProvider>
        {process.env.NODE_ENV === "production" && <Analytics />}
      </body>
    </html>
  );
}
