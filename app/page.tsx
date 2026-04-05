import { BoardApp } from "@/components/board-app";
import { loadYqh26DashboardData } from "@/lib/yqh26-data";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const storageMode = process.env.DATABASE_URL ? "remote" : "local";
  const dataset = await loadYqh26DashboardData();

  return <BoardApp storageMode={storageMode} dataset={dataset} />;
}
