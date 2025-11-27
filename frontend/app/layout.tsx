import "./globals.css";
import React from "react";
export const metadata = { title: "AI Policy Helper" };

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className=" antialiased bg-neutral-950">{children}</body>
    </html>
  );
}
