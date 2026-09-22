import "./globals.css";

export const metadata = {
  title: "ISRO Edge-AI Component Burn-In Screening & Mission Assurance",
  description:
    "Physics-Informed Edge AI System for High-Reliability Semiconductor Burn-In Screening: AEC-Q001 DPAT Anomaly Detection, 168h Time-Series Drift Forecasting, TreeSHAP Explainability, and Fail-Safe Hardware Relay Eject.",
  keywords: [
    "ISRO",
    "Semiconductor Screening",
    "Burn-In Testing",
    "Dynamic Part Average Testing",
    "DPAT",
    "XGBoost Drift Predictor",
    "Edge AI",
    "Raspberry Pi",
    "SHAP Explainability",
    "Space Mission Assurance",
  ],
  authors: [{ name: "Aerospace Edge-AI Engineering Team" }],
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
      <body>{children}</body>
    </html>
  );
}
