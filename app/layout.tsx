import type { Metadata } from "next";
import { CollaborationStrip } from "@/components/collaboration-strip";
import "./globals.css";

export const metadata: Metadata = {
  title: "YQuantum // Q-gars 2026",
  description: "Five-deck collaborative kanban board for the YQuantum team."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <CollaborationStrip />
        {children}
      </body>
    </html>
  );
}
