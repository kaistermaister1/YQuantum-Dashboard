import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YQuantum Board",
  description: "Five-deck collaborative kanban board for the YQuantum team."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
