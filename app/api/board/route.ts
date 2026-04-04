import { NextResponse } from "next/server";
import { hasDatabase, listCards } from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!hasDatabase()) {
    return NextResponse.json(
      {
        cards: [],
        storageMode: "local"
      },
      { status: 200 }
    );
  }

  try {
    const cards = await listCards();

    return NextResponse.json({
      cards,
      storageMode: "remote"
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to load board."
      },
      { status: 500 }
    );
  }
}
