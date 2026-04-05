// QUBO block landscape: wireframe height = Q_ij (Travelers hackathon data).
// Aesthetic matches kaistermaister1/c-physics: Raylib, black background, 3D lines, trackball.
// https://github.com/kaistermaister1/c-physics
//
// Planned “story mode” (multi-step narrative for DQI/QAOA demos): see README § Story mode.
// Future: UI buttons advance steps; each step loads or morphs fields (M, C, weights, penalties,
// full Q, block split, Hamiltonian / Ising view, QAOA circuit or bitstring output).

#include <raylib.h>
#include <raymath.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool OnScreen(const Vector2& p, int sw, int sh, int margin) {
    return p.x >= -margin && p.x <= sw + margin && p.y >= -margin && p.y <= sh + margin;
}

static bool PointInFrontOfCamera(Vector3 world, const Camera3D& cam) {
    Vector3 forward = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
    Vector3 to = Vector3Subtract(world, cam.position);
    return Vector3DotProduct(forward, to) > 0.05f;
}

struct QuboSurface {
    int n = 0;
    int nCoverage = 0;
    int nSlack = 0;
    int packageIndex = 0;
    double constantOffset = 0.0;
    std::vector<double> Q;
    std::vector<int> x;

    double qAt(int i, int j) const { return Q[static_cast<size_t>(i) * n + j]; }
};

static bool LoadQuboFile(const char* path, QuboSurface& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = std::string("cannot open ") + path;
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream head(line);
        if (!(head >> out.n >> out.nCoverage >> out.nSlack >> out.packageIndex >> out.constantOffset)) {
            err = "bad header line";
            return false;
        }
        out.Q.resize(static_cast<size_t>(out.n) * out.n);
        for (int i = 0; i < out.n; i++) {
            if (!std::getline(f, line)) {
                err = "unexpected EOF reading Q rows";
                return false;
            }
            std::istringstream row(line);
            for (int j = 0; j < out.n; j++) {
                double v;
                if (!(row >> v)) {
                    err = "bad Q row " + std::to_string(i);
                    return false;
                }
                out.Q[static_cast<size_t>(i) * out.n + j] = v;
            }
        }
        int xPresent = 0;
        if (!std::getline(f, line)) {
            err = "missing x_present line";
            return false;
        }
        {
            std::istringstream xs(line);
            if (!(xs >> xPresent)) {
                err = "bad x_present";
                return false;
            }
        }
        if (xPresent) {
            if (!std::getline(f, line)) {
                err = "missing x vector";
                return false;
            }
            std::istringstream xv(line);
            out.x.resize(static_cast<size_t>(out.n));
            for (int i = 0; i < out.n; i++) {
                int b;
                if (!(xv >> b)) {
                    err = "short x vector";
                    return false;
                }
                out.x[static_cast<size_t>(i)] = b ? 1 : 0;
            }
        }
        return true;
    }
    err = "empty file";
    return false;
}

static double MaxAbsQ(const QuboSurface& s) {
    double m = 0.0;
    for (double v : s.Q) m = std::max(m, std::abs(v));
    return m;
}

static double Energy(const QuboSurface& s) {
    double e = s.constantOffset;
    for (int i = 0; i < s.n; i++) {
        for (int j = 0; j < s.n; j++) {
            e += s.qAt(i, j) * static_cast<double>(s.x[static_cast<size_t>(i)]) *
                 static_cast<double>(s.x[static_cast<size_t>(j)]);
        }
    }
    return e;
}

static Color QHeatColor(double v, double maxAbs) {
    if (maxAbs <= 1e-18) return {90, 90, 95, 255};
    double t = v / maxAbs;
    if (t > 1.0) t = 1.0;
    if (t < -1.0) t = -1.0;
    if (t <= 0.0) {
        const float u = static_cast<float>(-t);
        return {
            (unsigned char)(35 + (220 * (1.0f - u))),
            (unsigned char)(45 + (210 * (1.0f - u))),
            255,
            255,
        };
    }
    const float u = static_cast<float>(t);
    return {
        255,
        (unsigned char)(35 + (210 * (1.0f - u))),
        (unsigned char)(45 + (200 * (1.0f - u))),
        255,
    };
}

struct QMatrixLayout {
    int ox = 16;
    int oy = 108;
    int cell = 8;
    int lm = 24;
    int th = 58;
    int barGap = 10;
    int barW = 18;
    int n = 0;
};

static QMatrixLayout MakeQMatrixLayout(const QuboSurface& surf, int sh) {
    QMatrixLayout L;
    L.n = surf.n;
    const int maxGrid = std::min(400, sh - 240);
    L.cell = std::max(4, std::min(30, maxGrid / std::max(1, surf.n)));
    return L;
}

static Rectangle QMatrixHitBounds(const QMatrixLayout& L) {
    const float gridW = static_cast<float>(L.lm + L.n * L.cell + 4);
    const float gridH = static_cast<float>(L.th + L.n * L.cell + 6);
    const float bar = static_cast<float>(L.barGap + L.barW + 16);
    return {static_cast<float>(L.ox), static_cast<float>(L.oy), gridW + bar, gridH + 28};
}

static void DrawQMatrixPanel(const QuboSurface& surf, const QMatrixLayout& L, double maxAbsQ, int sw,
                             int sh, Vector2 mouse) {
    int hoverI = -1;
    int hoverJ = -1;
    const int n = L.n;
    const int gx0 = L.ox + L.lm;
    const int gy0 = L.oy + L.th;
    const Rectangle outer = QMatrixHitBounds(L);
    DrawRectangleRec(outer, (Color){8, 10, 14, 235});
    DrawRectangleLinesEx(outer, 1, (Color){70, 85, 110, 255});

    char pqTitle[96];
    std::snprintf(pqTitle, sizeof(pqTitle), "Print Q — package %d block  (Q key)", surf.packageIndex);
    DrawText(pqTitle, L.ox + 8, L.oy + 4, 15, (Color){200, 210, 230, 255});
    DrawText("Same n x n Q as the 3D view (this package only).", L.ox + 8, L.oy + 22, 11,
             (Color){130, 200, 255, 255});
    DrawText("blue = negative   white ~ 0   red = positive", L.ox + 8, L.oy + 34, 11,
             (Color){130, 140, 155, 255});

    const int tickFs = (L.cell >= 14) ? 10 : 9;
    if (L.cell >= 11) {
        for (int j = 0; j < n; j++) {
            char lab[8];
            std::snprintf(lab, sizeof(lab), "%d", j);
            int tw = MeasureText(lab, tickFs);
            DrawText(lab, gx0 + j * L.cell + (L.cell - 1 - tw) / 2, gy0 - 12, tickFs, GRAY);
        }
    }

    for (int i = 0; i < n; i++) {
        char lab[8];
        std::snprintf(lab, sizeof(lab), "%d", i);
        DrawText(lab, L.ox + 2, gy0 + i * L.cell + std::max(0, (L.cell - tickFs) / 2), tickFs, GRAY);

        for (int j = 0; j < n; j++) {
            const double v = surf.qAt(i, j);
            const Color fill = QHeatColor(v, maxAbsQ);
            const float x = static_cast<float>(gx0 + j * L.cell);
            const float y = static_cast<float>(gy0 + i * L.cell);
            const Rectangle cellR = {x, y, static_cast<float>(L.cell) - 1.0f, static_cast<float>(L.cell) - 1.0f};
            DrawRectangleRec(cellR, fill);
            if (i == j) {
                DrawRectangleLinesEx(cellR, 1, (Color){160, 220, 255, 220});
            }
            if (CheckCollisionPointRec(mouse, cellR)) {
                hoverI = i;
                hoverJ = j;
            }
        }
    }

    const int bx = gx0 + n * L.cell + L.barGap;
    const int by = gy0;
    const int bh = n * L.cell;
    const int steps = 48;
    const int denom = std::max(1, steps - 1);
    for (int s = 0; s < steps; s++) {
        const double t = -1.0 + 2.0 * static_cast<double>(s) / static_cast<double>(denom);
        const Color c = QHeatColor(t * maxAbsQ, maxAbsQ);
        const float y0 = static_cast<float>(by + (s * bh) / steps);
        const float y1 = static_cast<float>(by + ((s + 1) * bh) / steps);
        DrawRectangle(bx, static_cast<int>(y0), L.barW, std::max(1, static_cast<int>(y1 - y0 + 0.5f)), c);
    }
    DrawRectangleLines(bx, by, L.barW, bh, (Color){120, 120, 130, 255});
    char zl[48];
    std::snprintf(zl, sizeof(zl), "%+.4g", maxAbsQ);
    DrawText(zl, bx + L.barW + 4, by - 2, 10, GRAY);
    std::snprintf(zl, sizeof(zl), "0");
    DrawText(zl, bx + L.barW + 4, by + bh / 2 - 5, 10, GRAY);
    std::snprintf(zl, sizeof(zl), "%+.4g", -maxAbsQ);
    DrawText(zl, bx + L.barW + 4, by + bh - 12, 10, GRAY);

    if (hoverI >= 0 && hoverJ >= 0) {
        const double hv = surf.qAt(hoverI, hoverJ);
        char tip[112];
        std::snprintf(tip, sizeof(tip), "Q[%d,%d] = %.8g", hoverI, hoverJ, hv);
        const int fs = 15;
        const int tw = MeasureText(tip, fs);
        const int padX = 10;
        const int padY = 8;
        const int boxW = tw + padX * 2;
        const int boxH = fs + padY * 2;
        int tx = static_cast<int>(mouse.x) + 16;
        int ty = static_cast<int>(mouse.y) + 16;
        if (tx + boxW + 8 > sw) tx = static_cast<int>(mouse.x) - boxW - 16;
        if (ty + boxH + 8 > sh) ty = static_cast<int>(mouse.y) - boxH - 16;
        DrawRectangle(tx - 2, ty - 2, boxW + 4, boxH + 4, (Color){0, 0, 0, 200});
        DrawRectangleLines(tx - 2, ty - 2, boxW + 4, boxH + 4, (Color){200, 200, 210, 255});
        DrawText(tip, tx + padX, ty + padY, fs, RAYWHITE);
    }
}

static void DrawLegendPanel(int sw, int sh, const QuboSurface& surf, bool showAssignmentKey) {
    const int pw = 432;
    const int px = sw - pw - 12;
    const int py = 72;
    const int ph = sh - py - 52;
    DrawRectangle(px - 6, py - 8, pw, ph, (Color){10, 12, 16, 230});
    DrawRectangleLines(px - 6, py - 8, pw, ph, (Color){55, 70, 90, 255});

    int y = py;
    const int fs = 13;
    const int lh = fs + 5;
    const Color title = {200, 210, 230, 255};
    const Color body = {175, 185, 200, 255};
    const Color accent = {120, 200, 255, 255};
    const Color gold = {255, 220, 100, 255};

    DrawText("What this view is", px, y, fs + 2, title);
    y += lh + 4;
    char scope[220];
    std::snprintf(scope, sizeof(scope),
                  "PACKAGE %d — ONE QUBO BLOCK only (column m of the bundle).", surf.packageIndex);
    DrawText(scope, px, y, fs, accent);
    y += lh;
    DrawText("Not the full (NM)x(NM) matrix; that is block-diagonal across packages.", px, y, fs, body);
    y += lh;
    DrawText("This block is symmetric n x n for variables of package m only.", px, y, fs, body);
    y += lh;
    DrawText("Energy (minimize):  E(x) = x' Q x + const", px, y, fs, body);
    y += lh;
    DrawText("Binary bits x_i in {0,1} (coverages + slacks).", px, y, fs, body);
    y += lh + 8;

    DrawText("The height field (gray mesh)", px, y, fs + 2, title);
    y += lh + 4;
    DrawText("Grid corner (i,j) is matrix entry Q_ij.", px, y, fs, body);
    y += lh;
    DrawText("Height = Q_ij scaled by max|Q| (shape, not raw units).", px, y, fs, body);
    y += lh;
    DrawText("Off-diagonal: coupling if BOTH x_i and x_j are 1.", px, y, fs, body);
    y += lh + 8;

    DrawText("Cyan pillars (labels at tips)", px, y, fs + 2, title);
    y += lh + 4;
    DrawText("Along the diagonal i = j: stem from y=0 to height Q_ii.", px, y, fs, accent);
    y += lh;
    DrawText("Q_ii acts like a linear term (x_i^2 = x_i).", px, y, fs, body);
    y += lh;
    DrawText("Profit shows up as negative diagonal; penalties add positive.", px, y, fs, body);
    y += lh + 8;

    DrawText("Valleys vs peaks", px, y, fs + 2, title);
    y += lh + 4;
    DrawText("Below the y=0 plane: negative Q_ij.", px, y, fs, body);
    y += lh;
    DrawText("  -> Lower energy when those bits turn on (favored).", px, y, fs, body);
    y += lh;
    DrawText("High ridges: large positive Q_ij.", px, y, fs, body);
    y += lh;
    DrawText("  -> Costly when paired bits are both 1 (penalties).", px, y, fs, body);
    y += lh + 8;

    if (showAssignmentKey) {
        DrawText("Gold markers", px, y, fs + 2, title);
        y += lh + 4;
        DrawText("Exported bitstring x_i = 1 at that variable.", px, y, fs, gold);
        y += lh + 8;
    }

    DrawText("Variable index (diagonal pillar = this row/col)", px, y, fs + 2, title);
    y += lh + 4;
    char row[96];
    for (int i = 0; i < surf.n; i++) {
        if (y > py + ph - lh - 6) {
            DrawText("(panel clipped — smaller n or taller window)", px, y, fs - 1, GRAY);
            break;
        }
        if (i < surf.nCoverage) {
            std::snprintf(row, sizeof(row), "  %2d   coverage bit  [%d]", i, i);
        } else {
            std::snprintf(row, sizeof(row), "  %2d   slack bit       [%d]", i, i - surf.nCoverage);
        }
        DrawText(row, px, y, fs - 1, body);
        y += lh - 1;
    }
}

static void StepTrackball(Camera3D& camera) {
    const float rotSpeed = 0.01f;
    const float panSpeed = 8.0f;
    const float zoomSpeed = 0.12f;
    const float minR = 0.05f;

    Vector3 offset = Vector3Subtract(camera.position, camera.target);
    if (Vector3Length(offset) < 0.0001f) offset = {0.0f, -5.0f, 0.0f};

    Vector3 up = camera.up;
    if (Vector3Length(up) < 0.0001f) up = {0.0f, 0.0f, 1.0f};
    up = Vector3Normalize(up);

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();

        Quaternion qYaw = QuaternionFromAxisAngle(up, -rotSpeed * d.x);
        offset = Vector3RotateByQuaternion(offset, qYaw);
        up = Vector3RotateByQuaternion(up, qYaw);

        Vector3 forward = Vector3Normalize(Vector3Negate(offset));
        Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, up));

        Quaternion qPitch = QuaternionFromAxisAngle(right, -rotSpeed * d.y);
        offset = Vector3RotateByQuaternion(offset, qPitch);
        up = Vector3RotateByQuaternion(up, qPitch);
    }

    float r = Vector3Length(offset);
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        r *= expf(-zoomSpeed * wheel);
        if (r < minR) r = minR;
        offset = Vector3Scale(Vector3Normalize(offset), r);
    }

    Vector3 forward = Vector3Normalize(Vector3Negate(offset));
    Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, up));
    up = Vector3Normalize(Vector3CrossProduct(right, forward));

    Vector3 move = {0.0f, 0.0f, 0.0f};
    if (IsKeyDown(KEY_W)) move = Vector3Add(move, up);
    if (IsKeyDown(KEY_S)) move = Vector3Subtract(move, up);
    if (IsKeyDown(KEY_D)) move = Vector3Add(move, right);
    if (IsKeyDown(KEY_A)) move = Vector3Subtract(move, right);

    if (Vector3Length(move) > 0.0001f) {
        move = Vector3Scale(Vector3Normalize(move), panSpeed * GetFrameTime());
        camera.target = Vector3Add(camera.target, move);
    }

    if (IsKeyPressed(KEY_SPACE)) camera.target = {0.0f, 0.0f, 0.0f};

    camera.position = Vector3Add(camera.target, offset);
    camera.up = up;
}

int main(int argc, char** argv) {
    const char* path = (argc >= 2) ? argv[1] : "qubo_surface.txt";
    QuboSurface surf;
    std::string err;
    if (!LoadQuboFile(path, surf, err)) {
        std::fprintf(stderr, "Load error: %s\n", err.c_str());
        return 1;
    }

    const int sw = 1400;
    const int sh = 900;
    char winTitle[96];
    std::snprintf(winTitle, sizeof(winTitle), "Package %d — QUBO block (single column, n=%d)",
                  surf.packageIndex, surf.n);
    InitWindow(sw, sh, winTitle);
    SetTargetFPS(120);

    Camera3D camera = {0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.position = {static_cast<float>(surf.n) * 0.9f, static_cast<float>(surf.n) * 0.55f,
                       static_cast<float>(surf.n) * 0.9f};
    camera.up = {0.0f, 1.0f, 0.0f};

    const double scaleDen = std::max(MaxAbsQ(surf), 1e-12);
    const float amp = static_cast<float>(std::max(surf.n, 3)) * 0.35f;
    const float gx = 1.0f;
    const float gz = 1.0f;
    const float ox = -(surf.n - 1) * gx * 0.5f;
    const float oz = -(surf.n - 1) * gz * 0.5f;

    auto gridPos = [&](int i, int j) -> Vector3 {
        double qij = surf.qAt(i, j);
        float y = static_cast<float>(qij / scaleDen) * amp;
        return {ox + i * gx, y, oz + j * gz};
    };

    bool autoSpin = true;
    bool showLegend = false;
    bool showQMatrix = false;

    while (!WindowShouldClose()) {
        const QMatrixLayout qLay = MakeQMatrixLayout(surf, sh);
        const Rectangle qHit = QMatrixHitBounds(qLay);
        const Vector2 mouse = GetMousePosition();
        const bool mouseOnQPanel = showQMatrix && CheckCollisionPointRec(mouse, qHit);

        if (!mouseOnQPanel) {
            StepTrackball(camera);
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && !mouseOnQPanel) autoSpin = false;
        if (IsKeyPressed(KEY_R)) autoSpin = !autoSpin;
        if (IsKeyPressed(KEY_I)) showLegend = !showLegend;
        if (IsKeyPressed(KEY_Q)) showQMatrix = !showQMatrix;

        if (autoSpin) {
            const Vector3 c = camera.target;
            const Vector3 p = camera.position;
            Vector3 off = {p.x - c.x, p.y - c.y, p.z - c.z};
            const float dt = GetFrameTime();
            const float ca = cosf(0.35f * dt);
            const float sa = sinf(0.35f * dt);
            const float nx = off.x * ca - off.z * sa;
            const float nz = off.x * sa + off.z * ca;
            camera.position = {c.x + nx, c.y + off.y, c.z + nz};
        }

        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode3D(camera);

        const Color lineCol = {220, 220, 220, 255};
        const Color diagCol = {120, 200, 255, 255};
        const Color assignCol = {255, 220, 100, 255};

        for (int i = 0; i < surf.n; i++) {
            for (int j = 0; j < surf.n; j++) {
                if (i + 1 < surf.n) {
                    DrawLine3D(gridPos(i, j), gridPos(i + 1, j), lineCol);
                }
                if (j + 1 < surf.n) {
                    DrawLine3D(gridPos(i, j), gridPos(i, j + 1), lineCol);
                }
                if (i == j) {
                    Vector3 p = gridPos(i, j);
                    DrawLine3D({p.x, 0.0f, p.z}, p, diagCol);
                }
            }
        }

        if (!surf.x.empty()) {
            for (int i = 0; i < surf.n; i++) {
                if (!surf.x[static_cast<size_t>(i)]) continue;
                Vector3 p = gridPos(i, i);
                float top = p.y + 0.35f * amp;
                DrawLine3D(p, {p.x, top, p.z}, assignCol);
                DrawSphere({p.x, top, p.z}, 0.08f * amp, assignCol);
            }
        }

        EndMode3D();

        const Color diagLabelCol = {120, 220, 255, 255};
        const int labelFs = 12;
        for (int i = 0; i < surf.n; i++) {
            Vector3 tip = gridPos(i, i);
            if (!PointInFrontOfCamera(tip, camera)) continue;
            Vector2 scr = GetWorldToScreen(tip, camera);
            if (!OnScreen(scr, sw, sh, 40)) continue;
            char tag[48];
            if (i < surf.nCoverage) {
                std::snprintf(tag, sizeof(tag), "%d  cov", i);
            } else {
                std::snprintf(tag, sizeof(tag), "%d  slk", i);
            }
            int tw = MeasureText(tag, labelFs);
            int tx = static_cast<int>(scr.x) - tw / 2;
            int ty = static_cast<int>(scr.y) - labelFs - 4;
            DrawText(tag, tx, ty, labelFs, diagLabelCol);
        }

        if (showLegend) {
            DrawLegendPanel(sw, sh, surf, !surf.x.empty());
        }

        if (showQMatrix) {
            DrawQMatrixPanel(surf, qLay, scaleDen, sw, sh, mouse);
        }

        char hudA[320];
        char hudB[320];
        std::snprintf(
            hudA, sizeof(hudA),
            "QUBO BLOCK — package %d  (this bundle column only; not full multi-package Q)",
            surf.packageIndex);
        std::snprintf(hudB, sizeof(hudB), "n=%d  (%d coverages + %d slacks)   max|Q|=%.4g   const=%.4g",
                      surf.n, surf.nCoverage, surf.nSlack, scaleDen, surf.constantOffset);
        DrawText(hudA, 20, 18, 17, RAYWHITE);
        DrawText(hudB, 20, 40, 15, (Color){180, 190, 205, 255});
        DrawText(
            "LMB: rotate   wheel: zoom   WASD: pan   SPACE: recenter   R: spin   I: info panel   Q: Print Q",
            20, sh - 36, 15, GRAY);
        DrawText(path, 20, 62, 15, GRAY);

        if (!surf.x.empty()) {
            char eh[128];
            std::snprintf(eh, sizeof(eh), "E(x) = x'Qx + const = %.6g", Energy(surf));
            DrawText(eh, 20, 82, 17, assignCol);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
