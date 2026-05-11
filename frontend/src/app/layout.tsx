import type { Metadata, Viewport } from "next";
import { ToastProvider } from "@/components/ToastProvider";
import { FeedbackWidget } from "@/components/FeedbackWidget";
import "./globals.css";

export const viewport: Viewport = {
  themeColor: "#00e5ff",
};

export const metadata: Metadata = {
  title: {
    default: "SignLearn — Real-time ASL ↔ English in your browser",
    template: "%s | SignLearn",
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Inline script: apply saved theme + text-size BEFORE first paint to avoid flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('sl-theme');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);else if(window.matchMedia('(prefers-color-scheme: light)').matches)document.documentElement.setAttribute('data-theme','light');var p=JSON.parse(localStorage.getItem('sl-prefs')||'{}');if(p.textSize==='large')document.documentElement.setAttribute('data-text-size','large');}catch(e){}})();`,
          }}
        />
      </head>
      <body>
        <ToastProvider>
          <a className="skip-nav" href="#main-content">
            Skip to main content
          </a>
          {children}
          <FeedbackWidget />
        </ToastProvider>
      </body>
    </html>
  );
}
