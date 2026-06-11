import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { Toaster } from "@/components/ui/toast";
import { TooltipProvider } from "@/components/ui/tooltip";
import { FeedbackWidget } from "@/components/FeedbackWidget";
import { OnboardingTour } from "@/components/primitives/OnboardingTour";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

const interDisplay = Inter({
  subsets: ["latin"],
  display: "swap",
  weight: ["500", "600", "700"],
  variable: "--font-inter-display",
});

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)",  color: "#0b1020" },
  ],
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: {
    default: "SignLearn — Real-time ASL ↔ English in your browser",
    template: "%s · SignLearn",
  },
  description:
    "Have a real conversation in American Sign Language and English. Real-time, in your browser, no app to install. Your video stays on your device.",
  metadataBase: new URL("https://signlearn.app"),
  openGraph: {
    type: "website",
    title: "SignLearn — Real-time ASL ↔ English",
    description: "Have a real conversation. No interpreter. No app. Just a link.",
    url: "https://signlearn.app",
    images: [{ url: "/og.png" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SignLearn — Real-time ASL ↔ English",
    description: "Have a real conversation. No interpreter. No app. Just a link.",
    images: ["/og.png"],
  },
  manifest: "/manifest.json",
  other: {
    "application/ld+json": JSON.stringify({
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      name: "SignLearn",
      applicationCategory: "AccessibilityApplication",
      operatingSystem: "Web",
      description:
        "Real-time American Sign Language to English translation in the browser.",
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    }),
  },
};

const FOUC_GUARD = `(function(){try{
  var p = JSON.parse(localStorage.getItem('signlearn.prefs.v1') || '{}');
  var h = document.documentElement;
  if (p.theme && p.theme !== 'system') h.setAttribute('data-theme', p.theme);
  if (p.textSize && p.textSize !== 'normal') h.setAttribute('data-text-size', p.textSize);
  if (p.reduceMotion) h.setAttribute('data-reduce-motion', 'true');
}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${interDisplay.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: FOUC_GUARD }} />
      </head>
      <body className="font-sans antialiased">
        <a className="skip-nav" href="#main-content">
          Skip to main content
        </a>
        <Toaster>
          <TooltipProvider delayDuration={300}>
            {children}
            <FeedbackWidget />
            <OnboardingTour />
          </TooltipProvider>
        </Toaster>
      </body>
    </html>
  );
}
