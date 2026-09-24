import "./globals.css";

export const metadata = {
  title: "PARIKSHAN-AI • ISRO Space-Grade Semiconductor Qualification & Burn-In Screening",
  description:
    "PARIKSHAN-AI: Physics-Informed Edge AI System for ISRO High-Reliability Semiconductor Burn-In Screening, AEC-Q001 DPAT Anomaly Detection, 168h Arrhenius PINN Drift Forecasting, and AS9100/MIL-STD-883 Digital Certification.",
  keywords: [
    "PARIKSHAN-AI",
    "PARIKSHAN",
    "ISRO",
    "Semiconductor Screening",
    "Burn-In Testing",
    "Dynamic Part Average Testing",
    "DPAT",
    "PINN",
    "Arrhenius Thermodynamics",
    "Conformal Prediction",
    "XGBoost Drift Predictor",
    "Edge AI",
    "Raspberry Pi",
    "SHAP Explainability",
    "AS9100 Rev D",
    "MIL-STD-883",
    "Space Mission Assurance",
  ],
  icons: {
    icon: "/parikshan_logo.png",
    apple: "/parikshan_logo.png",
  },
  authors: [{ name: "PARIKSHAN-AI Aerospace Engineering Team" }],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
