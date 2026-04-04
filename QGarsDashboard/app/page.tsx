import { BoardApp } from "@/components/board-app";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const storageMode = process.env.DATABASE_URL ? "remote" : "local";

  return <BoardApp storageMode={storageMode} />;
}
