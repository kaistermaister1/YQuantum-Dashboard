import { NextResponse } from "next/server";
import { deleteCard, hasDatabase, updateCard } from "@/lib/storage";
import { isDeckKey } from "@/lib/board";

type RouteContext = {
  params: Promise<{
    id: string;
  }>;
};

export async function PATCH(request: Request, context: RouteContext) {
  if (!hasDatabase()) {
    return NextResponse.json(
      { error: "Shared database mode is not configured yet." },
      { status: 503 }
    );
  }

  try {
    const { id } = await context.params;
    const body = await request.json();
    const { deck, title, notes, position } = body as {
      deck?: string;
      title?: string;
      notes?: string;
      position?: number;
    };

    if (deck !== undefined && !isDeckKey(deck)) {
      return NextResponse.json({ error: "Invalid deck." }, { status: 400 });
    }

    if (position !== undefined && typeof position !== "number") {
      return NextResponse.json({ error: "Invalid position." }, { status: 400 });
    }

    const card = await updateCard(id, {
      deck,
      title: title?.trim(),
      notes: notes?.trim(),
      position
    });

    if (!card) {
      return NextResponse.json({ error: "Card not found." }, { status: 404 });
    }

    return NextResponse.json({ card });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to update card."
      },
      { status: 500 }
    );
  }
}

export async function DELETE(_request: Request, context: RouteContext) {
  if (!hasDatabase()) {
    return NextResponse.json(
      { error: "Shared database mode is not configured yet." },
      { status: 503 }
    );
  }

  try {
    const { id } = await context.params;
    await deleteCard(id);

    return new NextResponse(null, { status: 204 });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to delete card."
      },
      { status: 500 }
    );
  }
}
