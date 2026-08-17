import { Geist } from "next/font/google";
import "./globals.css";

const geist = Geist({ subsets: ["latin"] });

export const metadata = {
  title: "FAQ Chatbot | E-Commerce Support",
  description:
    "AI-powered FAQ chatbot using TF-IDF and cosine similarity matching for e-commerce customer support.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={geist.className}>{children}</body>
    </html>
  );
}
