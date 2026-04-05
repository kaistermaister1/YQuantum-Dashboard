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
#include <cstdint>
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

/** Central palette: tweak here (or later load from config) so marble + mesh lines stay consistent. */
struct GraphStyle {
    Color meshLine{220, 220, 220, 255};       // Q wireframe grid (horizontal)
    Color diagonalLine{120, 200, 255, 255};  // diagonal pillars (cyan)
    Color assignmentLine{255, 220, 100, 255}; // gold assignment stems / markers
    Color marbleWire{175, 200, 230, 255};    // fallback marble wire if coverage idx out of range
    Color marbleFill{35, 42, 55, 90};        // if a>0: inner sphere uses family RGB with this alpha (story 1)
    Color marbleNeutralWire{250, 250, 252, 255};   // story 0: all marbles white wire before family tint
    Color marbleNeutralFill{210, 212, 220, 72};    // story 0: subtle fill (alpha; 0 = skip)
    Color storyFloorLine{55, 62, 78, 255};   // faint ground grid during marbles intro
    /** Dependency arrows: ``01_insurance_bundling.html`` ``.constraint-badge.dependency`` / ``--family-optional-2`` */
    Color depArrow{0, 166, 118, 255};
};

static GraphStyle g_style;

/** Per-coverage RGB from ``Travelers/docs/01_insurance_bundling.html`` (script ``FAMILIES`` colors +
 *  ``COVERAGES`` order = ``instance_coverages.csv`` rows 0..19). */
static Color CoverageFamilyColorBundlingHtml(int coverageIdx) {
    static const Color kTable[20] = {
        {227, 24, 55, 255},   // 0  auto_liability_basic      — auto_base #E31837
        {227, 24, 55, 255},   // 1  auto_liability_enhanced   — auto_base
        {194, 18, 48, 255},   // 2  homeowners               — property_base #C21230
        {194, 18, 48, 255},   // 3  condo_owners
        {194, 18, 48, 255},   // 4  renters
        {58, 123, 213, 255},  // 5  collision                — auto_physical #3a7bd5
        {58, 123, 213, 255},  // 6  comprehensive
        {0, 166, 118, 255},   // 7  medical_payments         — auto_medical #00a676
        {0, 166, 118, 255},   // 8  personal_injury_protection
        {230, 168, 23, 255},  // 9  uninsured_motorist       — motorist_protect #e6a817
        {230, 168, 23, 255},  // 10 underinsured_motorist
        {155, 89, 182, 255},  // 11 roadside_assistance      — auto_ancillary #9b59b6
        {155, 89, 182, 255},  // 12 rental_reimbursement
        {155, 89, 182, 255},  // 13 gap_insurance
        {230, 126, 34, 255},  // 14 personal_property_floater — property_addon #e67e22
        {230, 126, 34, 255},  // 15 additional_living_expense
        {26, 188, 156, 255},  // 16 personal_umbrella        — liability_extend #1abc9c
        {26, 188, 156, 255},  // 17 excess_liability
        {192, 57, 43, 255},   // 18 flood_insurance          — specialty_peril #c0392b
        {192, 57, 43, 255},   // 19 earthquake_insurance
    };
    if (coverageIdx < 0 || coverageIdx >= 20) {
        return g_style.marbleWire;
    }
    return kTable[coverageIdx];
}

/** Family id 0..8 matching ``01_insurance_bundling.html`` FAMILIES order (auto_base … specialty_peril). */
static int FamilyIndexForCoverage(int coverageIdx) {
    static const signed char kFam[20] = {
        0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8,
    };
    if (coverageIdx < 0 || coverageIdx >= 20) {
        return 0;
    }
    return kFam[coverageIdx];
}

static int CountSameFamilyBefore(int covIdx, int nCoverageMarbles) {
    const int fam = FamilyIndexForCoverage(covIdx);
    int cnt = 0;
    for (int k = 0; k < covIdx && k < nCoverageMarbles; k++) {
        if (FamilyIndexForCoverage(k) == fam) {
            cnt++;
        }
    }
    return cnt;
}

/** Hub (i,j) on the n×n floor for a **slot** 0..8 = row-major cell in the 3×3 anchor grid. */
static void FamilyHubIJ(int slotIdx, int n, int* outI, int* outJ) {
    const int g = 3;
    const int fc = slotIdx % g;
    const int fr = slotIdx / g;
    const int span = std::max(1, n - 1);
    *outI = std::clamp((2 * fc + 1) * span / (2 * g), 0, n - 1);
    *outJ = std::clamp((2 * fr + 1) * span / (2 * g), 0, n - 1);
}

/**
 * Maps semantic family index (``FamilyIndexForCoverage`` / ``01_insurance_bundling.html`` order) to a 3×3 **slot**
 * so families touched by ``instance_dependencies.csv`` sit next to each other — shorter arrows in story 2.
 * Layout (slot row-major): auto_base–liability on top-left pair; property_base center; specialty right;
 * property_addon below center; optional families fill corners.
 */
static int HubSlotForSemanticFamily(int semanticFam) {
    static const signed char kSlot[9] = {
        0,  // 0 auto_base          — next to liability (umbrella deps)
        4,  // 1 property_base      — center (homeowners / condo prerequisite hub)
        2,  // 2 auto_physical
        3,  // 3 auto_medical
        8,  // 4 motorist_protect
        6,  // 5 auto_ancillary
        7,  // 6 property_addon     — under property_base (floater, ALE)
        1,  // 7 liability_extend   — next to auto_base
        5,  // 8 specialty_peril    — next to property_base (flood / earthquake)
    };
    if (semanticFam < 0 || semanticFam >= 9) {
        return 0;
    }
    return static_cast<int>(kSlot[semanticFam]);
}

static void FamilySemanticHubIJ(int semanticFam, int n, int* outI, int* outJ) {
    FamilyHubIJ(HubSlotForSemanticFamily(semanticFam), n, outI, outJ);
}

/** Landed grid cell for this coverage: same family packs around one hub (adjacent cells on the Q grid). */
static void CoverageGroupLandXZ(int covIdx, int n, int nCoverageMarbles, float ox, float oz, float gx, float gz,
                                float* outX, float* outZ) {
    const int fam = FamilyIndexForCoverage(covIdx);
    int hubI = 0;
    int hubJ = 0;
    FamilySemanticHubIJ(fam, n, &hubI, &hubJ);
    const int rank = CountSameFamilyBefore(covIdx, nCoverageMarbles);
    const int di = (rank % 3) - 1;
    const int dj = ((rank / 3) % 3) - 1;
    const int ti = std::clamp(hubI + di, 0, n - 1);
    const int tj = std::clamp(hubJ + dj, 0, n - 1);
    *outX = ox + static_cast<float>(ti) * gx;
    *outZ = oz + static_cast<float>(tj) * gz;
}

/** Wider XZ layout around the same family hub so dependency arrows read clearly (float coords, not clamped to Q cells). */
static void CoverageSpreadLandXZ(int covIdx, int n, int nCoverageMarbles, float ox, float oz, float gx, float gz,
                                 float* outX, float* outZ) {
    const int fam = FamilyIndexForCoverage(covIdx);
    int hubI = 0;
    int hubJ = 0;
    FamilySemanticHubIJ(fam, n, &hubI, &hubJ);
    const int rank = CountSameFamilyBefore(covIdx, nCoverageMarbles);
    const float di = static_cast<float>((rank % 3) - 1);
    const float dj = static_cast<float>(((rank / 3) % 3) - 1);
    const float sep = 1.58f * gx;
    *outX = ox + static_cast<float>(hubI) * gx + di * sep;
    *outZ = oz + static_cast<float>(hubJ) * gz + dj * sep;
}

/** Edges from ``Travelers/docs/data/YQH26_data/instance_dependencies.csv``: required_coverage → dependent_coverage. */
struct CoverageDependencyEdge {
    int requiredIdx;
    int dependentIdx;
};

static constexpr CoverageDependencyEdge kCoverageDependencyEdges[] = {
    {2, 15},  // homeowners → additional_living_expense
    {2, 18},  // homeowners → flood_insurance
    {2, 19},  // homeowners → earthquake_insurance
    {1, 16},  // auto_liability_enhanced → personal_umbrella
    {1, 17},  // auto_liability_enhanced → excess_liability
    {3, 14},  // condo_owners → personal_property_floater
};

static void DrawArrow3D(Vector3 a, Vector3 b, Color c) {
    Vector3 d = Vector3Subtract(b, a);
    const float len = Vector3Length(d);
    if (len < 1e-5f) {
        return;
    }
    Vector3 dir = Vector3Scale(d, 1.f / len);
    float headLen = len * 0.28f;
    if (headLen > 0.24f * len) {
        headLen = 0.24f * len;
    }
    if (headLen < 0.06f) {
        headLen = std::min(0.06f, len * 0.5f);
    }
    Vector3 neck = Vector3Subtract(b, Vector3Scale(dir, headLen));

    DrawLine3D(a, neck, c);

    Vector3 up = {0.f, 1.f, 0.f};
    if (std::fabs(dir.y) > 0.92f) {
        up = {1.f, 0.f, 0.f};
    }
    Vector3 side = Vector3Normalize(Vector3CrossProduct(up, dir));
    const float hw = headLen * 0.55f;
    Vector3 wing1 = Vector3Add(neck, Vector3Scale(side, hw));
    Vector3 wing2 = Vector3Subtract(neck, Vector3Scale(side, hw));
    DrawLine3D(b, wing1, c);
    DrawLine3D(b, wing2, c);
}

static void DrawDependencyArrowBetweenMarbles(Vector3 centerFrom, Vector3 centerTo, float marbleR, Color c) {
    Vector3 d = Vector3Subtract(centerTo, centerFrom);
    const float len = Vector3Length(d);
    const float inset = marbleR * 1.08f;
    if (len < 2.f * inset + 1e-4f) {
        return;
    }
    Vector3 dir = Vector3Scale(d, 1.f / len);
    Vector3 a = Vector3Add(centerFrom, Vector3Scale(dir, inset));
    Vector3 b = Vector3Subtract(centerTo, Vector3Scale(dir, inset));
    DrawArrow3D(a, b, c);
}

static float SmoothStep(float t) {
    if (t <= 0.f) {
        return 0.f;
    }
    if (t >= 1.f) {
        return 1.f;
    }
    return t * t * (3.f - 2.f * t);
}

static float EaseInOutCubic(float t) {
    if (t <= 0.f) {
        return 0.f;
    }
    if (t >= 1.f) {
        return 1.f;
    }
    if (t < 0.5f) {
        return 4.f * t * t * t;
    }
    const float u = -2.f * t + 2.f;
    return 1.f - (u * u * u) / 2.f;
}

static Color LerpColorRgb(const Color& a, const Color& b, float t) {
    if (t <= 0.f) {
        return a;
    }
    if (t >= 1.f) {
        return b;
    }
    auto l = [t](unsigned char u, unsigned char v) -> unsigned char {
        return static_cast<unsigned char>(
            std::round(static_cast<float>(u) + t * (static_cast<float>(v) - static_cast<float>(u))));
    };
    return {l(a.r, b.r), l(a.g, b.g), l(a.b, b.b), 255};
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

enum class StoryStep : int {
    MarblesDropWhite = 0,   // coverages fall as neutral (white) marbles
    MarblesFamilyColor = 1,  // blend to family colors, pause, then slide to family clusters (→ from step 0)
    MarblesDependencies = 2,  // spread for clarity + arrows from instance_dependencies.csv (→ from step 1)
    QBlockField = 3,
    Count = 4,
};

static constexpr float kFamilyColorBlendSec = 0.85f;
static constexpr float kFamilyPauseAfterColorSec = 0.5f;
static constexpr float kFamilyGroupMoveSec = 1.15f;
static constexpr float kDepSpreadSec = 0.95f;

static float EaseOutCubic(float t) {
    if (t <= 0.f) return 0.f;
    if (t >= 1.f) return 1.f;
    const float u = 1.f - t;
    return 1.f - u * u * u;
}

static void DrawStoryFloorGrid(float ox, float oz, float gx, float gz, int n, const Color& col) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const float x0 = ox + i * gx;
            const float z0 = oz + j * gz;
            if (i + 1 < n) {
                DrawLine3D({x0, 0.f, z0}, {ox + (i + 1) * gx, 0.f, z0}, col);
            }
            if (j + 1 < n) {
                DrawLine3D({x0, 0.f, z0}, {x0, 0.f, oz + (j + 1) * gz}, col);
            }
        }
    }
}

/** Landing (x,z) on the floor for Q row/col index `varIdx` — scattered, deterministic (same every run). */
static void MarbleLandXZ(int varIdx, int n, float ox, float oz, float gx, float gz, float* outX, float* outZ) {
    const uint32_t hx = static_cast<uint32_t>(varIdx) * 2654435761u ^ 0xA5A5A5A5u;
    const uint32_t hz = static_cast<uint32_t>(varIdx) * 2246822519u ^ 0x5BD1E995u;
    const float spanX = static_cast<float>(std::max(0, n - 1)) * gx;
    const float spanZ = static_cast<float>(std::max(0, n - 1)) * gz;
    const float margin = 0.1f;
    const float tx = (hx / float(UINT32_MAX)) * (1.f - 2.f * margin) + margin;
    const float tz = (hz / float(UINT32_MAX)) * (1.f - 2.f * margin) + margin;
    *outX = ox + tx * spanX;
    *outZ = oz + tz * spanZ;
}

/** Fixed-seed Fisher–Yates: drop order is pseudo-random but identical on every launch for the same `n`. */
static void BuildFixedMarbleDropOrder(int n, std::vector<int>& order) {
    order.resize(static_cast<size_t>(std::max(0, n)));
    for (int i = 0; i < n; i++) {
        order[static_cast<size_t>(i)] = i;
    }
    uint64_t s = 0xCAFEBABEC0DEF00Dull ^ (static_cast<uint64_t>(std::max(1, n)) * 1315423911ull);
    for (int i = n - 1; i > 0; i--) {
        s = s * 6364136223846793005ull + 1ull;
        const int j = static_cast<int>(s % static_cast<uint64_t>(i + 1));
        std::swap(order[static_cast<size_t>(i)], order[static_cast<size_t>(j)]);
    }
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

    const int nCoverageMarbles = std::max(0, std::min(surf.nCoverage, surf.n));
    std::vector<int> marbleDropOrder;
    BuildFixedMarbleDropOrder(nCoverageMarbles, marbleDropOrder);

    bool autoSpin = false;
    bool showLegend = false;
    bool showQMatrix = false;

    StoryStep storyStep = StoryStep::MarblesDropWhite;
    double marbleAnimT0 = GetTime();
    float marbleFamilyPhaseElapsed = 0.f;
    bool marbleFamilyPlayEntrance = true;
    float depSpreadElapsed = 0.f;
    bool depPlayEntrance = true;

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

        const StoryStep prevStory = storyStep;
        if (IsKeyPressed(KEY_RIGHT)) {
            const int next = std::min(static_cast<int>(storyStep) + 1, static_cast<int>(StoryStep::Count) - 1);
            storyStep = static_cast<StoryStep>(next);
        }
        if (IsKeyPressed(KEY_LEFT)) {
            const int next = std::max(static_cast<int>(storyStep) - 1, 0);
            storyStep = static_cast<StoryStep>(next);
        }
        if (storyStep == StoryStep::MarblesDropWhite && prevStory != StoryStep::MarblesDropWhite) {
            marbleAnimT0 = GetTime();
        }
        if (storyStep != prevStory && storyStep == StoryStep::MarblesFamilyColor) {
            marbleFamilyPlayEntrance = (prevStory == StoryStep::MarblesDropWhite);
            marbleFamilyPhaseElapsed = 0.f;
        }
        if (storyStep != prevStory && storyStep == StoryStep::MarblesDependencies) {
            depPlayEntrance = (prevStory == StoryStep::MarblesFamilyColor);
            depSpreadElapsed = 0.f;
        }

        if (storyStep != StoryStep::MarblesFamilyColor) {
            marbleFamilyPhaseElapsed = 0.f;
        } else if (marbleFamilyPlayEntrance) {
            marbleFamilyPhaseElapsed += GetFrameTime();
        }

        if (storyStep != StoryStep::MarblesDependencies) {
            depSpreadElapsed = 0.f;
        } else if (depPlayEntrance) {
            depSpreadElapsed += GetFrameTime();
        }

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

        if (storyStep == StoryStep::MarblesDropWhite || storyStep == StoryStep::MarblesFamilyColor ||
            storyStep == StoryStep::MarblesDependencies) {
            DrawStoryFloorGrid(ox, oz, gx, gz, surf.n, g_style.storyFloorLine);

            const int nMarbles = nCoverageMarbles;
            const float startY = amp * 9.0f;
            const float landY = amp * 0.14f;
            const float marbleR = 0.052f * amp;
            const double now = GetTime();
            const float rawT = static_cast<float>(now - marbleAnimT0);
            const bool colorByFamily = (storyStep == StoryStep::MarblesFamilyColor ||
                                        storyStep == StoryStep::MarblesDependencies);

            float depSpreadT = 0.f;
            if (storyStep == StoryStep::MarblesDependencies) {
                if (!depPlayEntrance) {
                    depSpreadT = 1.f;
                } else {
                    depSpreadT = EaseInOutCubic(std::min(1.f, depSpreadElapsed / kDepSpreadSec));
                }
            }

            float familyColorBlend = 0.f;
            float familyGroupMove = 0.f;
            if (colorByFamily) {
                if (storyStep == StoryStep::MarblesDependencies) {
                    familyColorBlend = 1.f;
                    familyGroupMove = 1.f;
                } else if (!marbleFamilyPlayEntrance) {
                    familyColorBlend = 1.f;
                    familyGroupMove = 1.f;
                } else {
                    const float el = marbleFamilyPhaseElapsed;
                    if (el < kFamilyColorBlendSec) {
                        familyColorBlend = SmoothStep(el / kFamilyColorBlendSec);
                        familyGroupMove = 0.f;
                    } else if (el < kFamilyColorBlendSec + kFamilyPauseAfterColorSec) {
                        familyColorBlend = 1.f;
                        familyGroupMove = 0.f;
                    } else {
                        familyColorBlend = 1.f;
                        const float tm =
                            (el - kFamilyColorBlendSec - kFamilyPauseAfterColorSec) / kFamilyGroupMoveSec;
                        familyGroupMove = EaseInOutCubic(std::min(1.f, tm));
                    }
                }
            }

            std::vector<Vector3> marbleCenterForCov(static_cast<size_t>(std::max(1, nMarbles)));

            for (int dropSlot = 0; dropSlot < nMarbles; dropSlot++) {
                const size_t ord = static_cast<size_t>(dropSlot);
                const int covIdx =
                    (ord < marbleDropOrder.size()) ? marbleDropOrder[ord] : dropSlot;
                float scatterX = 0.f;
                float scatterZ = 0.f;
                MarbleLandXZ(covIdx, surf.n, ox, oz, gx, gz, &scatterX, &scatterZ);

                float lx = scatterX;
                float lz = scatterZ;
                if (storyStep == StoryStep::MarblesDependencies) {
                    float cx = 0.f;
                    float cz = 0.f;
                    float sx = 0.f;
                    float sz = 0.f;
                    CoverageGroupLandXZ(covIdx, surf.n, nMarbles, ox, oz, gx, gz, &cx, &cz);
                    CoverageSpreadLandXZ(covIdx, surf.n, nMarbles, ox, oz, gx, gz, &sx, &sz);
                    lx = cx + (sx - cx) * depSpreadT;
                    lz = cz + (sz - cz) * depSpreadT;
                } else if (colorByFamily) {
                    float groupX = scatterX;
                    float groupZ = scatterZ;
                    CoverageGroupLandXZ(covIdx, surf.n, nMarbles, ox, oz, gx, gz, &groupX, &groupZ);
                    lx = scatterX + (groupX - scatterX) * familyGroupMove;
                    lz = scatterZ + (groupZ - scatterZ) * familyGroupMove;
                }

                float e = 1.f;
                if (!colorByFamily) {
                    const float stagger = 0.11f * static_cast<float>(dropSlot);
                    float u = (rawT * 0.52f) - stagger;
                    if (u < 0.f) u = 0.f;
                    if (u > 1.f) u = 1.f;
                    e = EaseOutCubic(u);
                }
                const float y = startY + (landY - startY) * e;
                const Vector3 marbleCenter = {lx, y, lz};
                if (covIdx >= 0 && covIdx < nMarbles) {
                    marbleCenterForCov[static_cast<size_t>(covIdx)] = marbleCenter;
                }

                const Color fam = CoverageFamilyColorBundlingHtml(covIdx);
                Color wire = g_style.marbleNeutralWire;
                Color fillRgb = {g_style.marbleNeutralFill.r, g_style.marbleNeutralFill.g, g_style.marbleNeutralFill.b,
                                 255};
                unsigned char fillAlpha = g_style.marbleNeutralFill.a;
                if (colorByFamily) {
                    wire = LerpColorRgb(g_style.marbleNeutralWire, fam, familyColorBlend);
                    fillRgb = LerpColorRgb(
                        {g_style.marbleNeutralFill.r, g_style.marbleNeutralFill.g, g_style.marbleNeutralFill.b, 255},
                        fam, familyColorBlend);
                    fillAlpha = static_cast<unsigned char>(std::round(
                        static_cast<float>(g_style.marbleNeutralFill.a) +
                        familyColorBlend *
                            static_cast<float>(static_cast<int>(g_style.marbleFill.a) -
                                               static_cast<int>(g_style.marbleNeutralFill.a))));
                }
                if (fillAlpha > 0) {
                    Color fillTint = fillRgb;
                    fillTint.a = fillAlpha;
                    DrawSphere(marbleCenter, marbleR * 0.88f, fillTint);
                }
                DrawSphereWires(marbleCenter, marbleR, 8, 12, wire);
            }

            if (storyStep == StoryStep::MarblesDependencies && depSpreadT > 0.02f && nMarbles > 0) {
                Color ac = g_style.depArrow;
                ac.a = static_cast<unsigned char>(
                    std::round(static_cast<float>(ac.a) *
                               std::min(1.f, depSpreadT * 1.05f)));
                for (const CoverageDependencyEdge& de : kCoverageDependencyEdges) {
                    if (de.requiredIdx < 0 || de.dependentIdx < 0 || de.requiredIdx >= nMarbles ||
                        de.dependentIdx >= nMarbles) {
                        continue;
                    }
                    DrawDependencyArrowBetweenMarbles(marbleCenterForCov[static_cast<size_t>(de.requiredIdx)],
                                                      marbleCenterForCov[static_cast<size_t>(de.dependentIdx)],
                                                      marbleR, ac);
                }
            }
        } else {
            for (int i = 0; i < surf.n; i++) {
                for (int j = 0; j < surf.n; j++) {
                    if (i + 1 < surf.n) {
                        DrawLine3D(gridPos(i, j), gridPos(i + 1, j), g_style.meshLine);
                    }
                    if (j + 1 < surf.n) {
                        DrawLine3D(gridPos(i, j), gridPos(i, j + 1), g_style.meshLine);
                    }
                    if (i == j) {
                        Vector3 p = gridPos(i, j);
                        DrawLine3D({p.x, 0.0f, p.z}, p, g_style.diagonalLine);
                    }
                }
            }

            if (!surf.x.empty()) {
                for (int i = 0; i < surf.n; i++) {
                    if (!surf.x[static_cast<size_t>(i)]) continue;
                    Vector3 p = gridPos(i, i);
                    float top = p.y + 0.35f * amp;
                    DrawLine3D(p, {p.x, top, p.z}, g_style.assignmentLine);
                    DrawSphere({p.x, top, p.z}, 0.08f * amp, g_style.assignmentLine);
                }
            }
        }

        EndMode3D();

        if (storyStep == StoryStep::QBlockField) {
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
            "LMB: rotate   wheel: zoom   WASD: pan   SPACE: recenter   R: spin   I: info   Q: Print Q   "
            "<- / -> : story",
            20, sh - 36, 15, GRAY);
        DrawText(path, 20, 62, 15, GRAY);

        {
            const char* stepLabel = "";
            if (storyStep == StoryStep::MarblesDropWhite) {
                stepLabel = "Story 0: coverages drop (white) — press → for family colors";
            } else if (storyStep == StoryStep::MarblesFamilyColor) {
                stepLabel =
                    "Story 1: tint → pause → slide to family clusters — → for dependency arrows "
                    "(YQH26_data/instance_dependencies.csv)";
            } else if (storyStep == StoryStep::MarblesDependencies) {
                stepLabel =
                    "Story 2: spread + requires→dependent arrows (bundling doc palette) — → for Q block";
            } else {
                stepLabel = "Story 3: Q block height field (matrix Q_ij)";
            }
            DrawText(stepLabel, 20, 104, 15, (Color){140, 200, 255, 255});
        }

        if (storyStep == StoryStep::QBlockField && !surf.x.empty()) {
            char eh[128];
            std::snprintf(eh, sizeof(eh), "E(x) = x'Qx + const = %.6g", Energy(surf));
            DrawText(eh, 20, 126, 17, g_style.assignmentLine);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
