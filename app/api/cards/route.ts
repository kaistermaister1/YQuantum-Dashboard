import { NextResponse } from "next/server";
import { createCard, hasDatabase } from "@/lib/storage";
import { isDeckKey } from "@/lib/board";

export async function POST(request: Request) {
  if (!hasDatabase()) {
    return NextResponse.json(
      { error: "Shared database mode is not configured yet." },
      { status: 503 }
    );
  }

  try {
    const body = await request.json();
    const { id, deck, title, notes, position } = body as {
      id?: string;
      deck?: string;
      title?: string;
      notes?: string;
      position?: number;
    };

    if (!id || !deck || !isDeckKey(deck) || !title?.trim() || typeof position !== "number") {
      return NextResponse.json({ error: "Invalid card payload." }, { status: 400 });
    }

    const card = await createCard({
      id,
      deck,
      title: title.trim(),
      notes: notes?.trim() ?? "",
      position
    });

    return NextResponse.json({ card }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to create card."
      },
      { status: 500 }
    );
  }
}
