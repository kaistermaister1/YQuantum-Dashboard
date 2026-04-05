// QUBO block landscape: wireframe height = Q_ij (Travelers hackathon data).
// Aesthetic matches kaistermaister1/c-physics: Raylib, black background, 3D lines, trackball.
// https://github.com/kaistermaister1/c-physics
//
// Planned “story mode” (multi-step narrative for DQI/QAOA demos): see README § Story mode.
// Future: UI buttons advance steps; each step loads or morphs fields (M, C, weights, penalties,
// full Q, block split, Hamiltonian / Ising view, QAOA circuit or bitstring output).

// Raylib’s raymath uses `{ 0 }` on nested structs; that trips -Wextra in our TU. Suppress only around headers.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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

    /** If true, ``Q_ij = marginScale * qMarginDiag[i] (on diag) + lambdaLive * qPen``; const = lambdaLive * constPerLambda. */
    bool parametricLambda = false;
    double lambdaLive = 0.0;
    double constPerLambda = 0.0;
    std::vector<double> qMarginDiag;
    std::vector<double> qPen;

    double qAt(int i, int j) const { return Q[static_cast<size_t>(i) * n + j]; }
};

static void ApplyLambdaToQ(QuboSurface& s, double marginScale) {
    if (!s.parametricLambda || s.n <= 0) {
        return;
    }
    const int n = s.n;
    const double ms = std::max(1e-18, marginScale);
    s.Q.resize(static_cast<size_t>(n) * n);
    s.constantOffset = s.lambdaLive * s.constPerLambda;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const double margin = (i == j) ? s.qMarginDiag[static_cast<size_t>(i)] : 0.0;
            s.Q[static_cast<size_t>(i) * n + j] =
                margin * ms + s.lambdaLive * s.qPen[static_cast<size_t>(i) * n + j];
        }
    }
}

static constexpr float kLambdaSliderW = 400.f;
static constexpr float kLambdaSliderH = 16.f;

static float SliderTFromLogLambda(double lam, double logMin, double logMax) {
    const double t = (std::log10(std::max(lam, 1e-30)) - logMin) / (logMax - logMin);
    return static_cast<float>(std::max(0.0, std::min(1.0, t)));
}

static double LambdaFromSliderT(float t, double logMin, double logMax) {
    t = std::max(0.f, std::min(1.f, t));
    return std::pow(10.0, logMin + static_cast<double>(t) * (logMax - logMin));
}

static Rectangle ParamSliderHitRect(float x, float y) {
    return {x - 8.f, y - 12.f, kLambdaSliderW + 16.f, kLambdaSliderH + 24.f};
}

/** QAOA accent #0066CC; track matches app shell grays. */
static void DrawLambdaPenaltySlider(float x, float y, double lam, double logMin, double logMax) {
    const Rectangle track = {x, y, kLambdaSliderW, kLambdaSliderH};
    DrawRectangleRec(track, (Color){38, 44, 56, 245});
    DrawRectangleLinesEx(track, 1.f, (Color){80, 140, 165, 220});
    const float t = SliderTFromLogLambda(lam, logMin, logMax);
    const float knobW = 11.f;
    const float knobH = kLambdaSliderH + 10.f;
    float knobX = x + t * kLambdaSliderW - knobW * 0.5f;
    knobX = std::max(x, std::min(x + kLambdaSliderW - knobW, knobX));
    const Rectangle knob = {knobX, y - 5.f, knobW, knobH};
    DrawRectangleRounded(knob, 3.f, 8, (Color){0, 102, 204, 255});
    DrawRectangleRoundedLines(knob, 3.f, 8, (Color){160, 220, 255, 255});
}

/** DQI blue #00356B for margin-scale knob. */
static void DrawMarginScaleSlider(float x, float y, double s, double logMin, double logMax) {
    const Rectangle track = {x, y, kLambdaSliderW, kLambdaSliderH};
    DrawRectangleRec(track, (Color){38, 44, 56, 245});
    DrawRectangleLinesEx(track, 1.f, (Color){70, 110, 150, 220});
    const float t = SliderTFromLogLambda(s, logMin, logMax);
    const float knobW = 11.f;
    const float knobH = kLambdaSliderH + 10.f;
    float knobX = x + t * kLambdaSliderW - knobW * 0.5f;
    knobX = std::max(x, std::min(x + kLambdaSliderW - knobW, knobX));
    const Rectangle knob = {knobX, y - 5.f, knobW, knobH};
    DrawRectangleRounded(knob, 3.f, 8, (Color){0, 53, 107, 255});
    DrawRectangleRoundedLines(knob, 3.f, 8, (Color){160, 200, 240, 255});
}

static bool ReadDoublesLine(const std::string& line, std::vector<double>* out) {
    out->clear();
    std::istringstream ls(line);
    double v;
    while (ls >> v) {
        out->push_back(v);
    }
    return !out->empty();
}

static bool LoadQuboFile(const char* path, QuboSurface& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = std::string("cannot open ") + path;
        return false;
    }
    out.parametricLambda = false;
    out.qMarginDiag.clear();
    out.qPen.clear();
    out.x.clear();

    std::string line;
    std::vector<double> hdr;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!ReadDoublesLine(line, &hdr)) {
            continue;
        }
        if (hdr.size() < 5u) {
            err = "bad header (need at least 5 numbers)";
            return false;
        }
        out.n = static_cast<int>(hdr[0]);
        out.nCoverage = static_cast<int>(hdr[1]);
        out.nSlack = static_cast<int>(hdr[2]);
        out.packageIndex = static_cast<int>(hdr[3]);
        if (hdr.size() >= 6u) {
            out.parametricLambda = true;
            out.constPerLambda = hdr[4];
            out.lambdaLive = hdr[5];
        } else {
            out.constantOffset = hdr[4];
        }
        break;
    }
    if (hdr.size() < 5u) {
        err = "empty file or no header";
        return false;
    }
    if (out.n < 1) {
        err = "invalid n";
        return false;
    }

    if (out.parametricLambda) {
        if (!std::getline(f, line)) {
            err = "v2: missing margin diagonal line";
            return false;
        }
        std::vector<double> diag;
        if (!ReadDoublesLine(line, &diag) || static_cast<int>(diag.size()) != out.n) {
            err = "v2: bad margin diagonal (need n floats)";
            return false;
        }
        out.qMarginDiag = std::move(diag);
        out.qPen.resize(static_cast<size_t>(out.n) * out.n);
        for (int i = 0; i < out.n; i++) {
            if (!std::getline(f, line)) {
                err = "v2: unexpected EOF reading Q_pen";
                return false;
            }
            std::istringstream row(line);
            for (int j = 0; j < out.n; j++) {
                double v;
                if (!(row >> v)) {
                    err = "v2: bad Q_pen row " + std::to_string(i);
                    return false;
                }
                out.qPen[static_cast<size_t>(i) * out.n + j] = v;
            }
        }
        out.Q.resize(static_cast<size_t>(out.n) * out.n);
        ApplyLambdaToQ(out, 1.0);
    } else {
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

static double MaxAbsQ(const QuboSurface& s) {
    double m = 0.0;
    for (double v : s.Q) m = std::max(m, std::abs(v));
    return m;
}

static double MaxAbsQSubmatrix(const QuboSurface& s, int bi, int bj, int qn) {
    double m = 0.0;
    for (int i = 0; i < qn; i++) {
        for (int j = 0; j < qn; j++) {
            const int gi = bi + i;
            const int gj = bj + j;
            if (gi >= 0 && gi < s.n && gj >= 0 && gj < s.n) {
                m = std::max(m, std::abs(s.qAt(gi, gj)));
            }
        }
    }
    return m;
}

static bool SurfHasAssignment(const QuboSurface& s) {
    return s.x.size() == static_cast<size_t>(s.n);
}

/** True if every variable index that appears in the qn×qn block (rows bi+*, cols bj+*) is selected (x=1). */
static bool QBlockVarsAllOne(const QuboSurface& s, int bi, int bj, int qn) {
    if (!SurfHasAssignment(s)) {
        return false;
    }
    for (int i = 0; i < qn; i++) {
        const int gi = bi + i;
        if (gi < 0 || gi >= s.n || !s.x[static_cast<size_t>(gi)]) {
            return false;
        }
    }
    for (int j = 0; j < qn; j++) {
        const int gj = bj + j;
        if (gj < 0 || gj >= s.n || !s.x[static_cast<size_t>(gj)]) {
            return false;
        }
    }
    return true;
}

/**
 * Max |Q_ij x_i x_j| on the block for mesh scaling during the Hamiltonian morph.
 * If ``useAlternatingDemoBits``, uses x_i=(gi mod 2) so terms vary when real x is missing or all-1 (otherwise
 * Q_ij·x_i·x_j == Q_ij and the height morph is invisible).
 */
static double MaxAbsQEnergyTermSubmatrixForMorph(const QuboSurface& s, int bi, int bj, int qn,
                                                 bool useAlternatingDemoBits) {
    double m = 0.0;
    for (int ii = 0; ii < qn; ii++) {
        for (int jj = 0; jj < qn; jj++) {
            const int gi = bi + ii;
            const int gj = bj + jj;
            if (gi < 0 || gj < 0 || gi >= s.n || gj >= s.n) {
                continue;
            }
            double xi;
            double xj;
            if (useAlternatingDemoBits) {
                xi = static_cast<double>(gi & 1);
                xj = static_cast<double>(gj & 1);
            } else {
                xi = static_cast<double>(s.x[static_cast<size_t>(gi)]);
                xj = static_cast<double>(s.x[static_cast<size_t>(gj)]);
            }
            const double t = s.qAt(gi, gj) * xi * xj;
            m = std::max(m, std::abs(t));
        }
    }
    return m;
}

/** One extracted square block from the loaded Q (for a smaller 3D demo mesh). */
static constexpr int kQDemoBlockN = 13;

static void ComputeDemoQBlock(const QuboSurface& surf, int* outBaseI, int* outBaseJ, int* outQn) {
    const int cap = std::min(kQDemoBlockN, std::max(1, surf.n));
    int qn = cap;
    if (surf.n <= qn) {
        *outBaseI = 0;
        *outBaseJ = 0;
        *outQn = surf.n;
        return;
    }
    const int nTile = (surf.n + qn - 1) / qn;
    const int selTI = nTile / 2;
    const int selTJ = nTile / 2;
    int bi = selTI * qn;
    int bj = selTJ * qn;
    bi = std::max(0, std::min(bi, surf.n - qn));
    bj = std::max(0, std::min(bj, surf.n - qn));
    *outBaseI = bi;
    *outBaseJ = bj;
    *outQn = qn;
}

/** Multiply only upward (positive) mesh heights; downward valleys keep baseline scale (Stories 7–9). */
static constexpr float kQBlockPositivePeakGain = 2.25f;

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
    /** Incompatibility links: ``.constraint-badge.incompatible`` / ``--family-optional-7`` (#c0392b) */
    Color incompatLine{192, 57, 43, 255};
    Color bundlerTray{88, 95, 110, 255};
    Color bundlerTrayWire{140, 155, 175, 255};
    Color checkAccent{45, 200, 120, 255};
    Color bundleWrap{100, 170, 230, 90};
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

/**
 * From ``instance_coverages.csv`` ``mandatory``: ``True`` = mandatory family (exactly one base choice);
 * ``False`` = optional add-on. YQH26 explicit incompatible pairs are all optional↔optional; we skip drawing
 * any pair involving a mandatory base bit so the viz matches “incompatibilities between add-ons.”
 */
static bool CoverageIsMandatoryFamilyPick(int coverageIdx) {
    static const bool kMandatory[20] = {
        true,  true,  true,  true,  true,   // 0-4  auto_base / property_base
        false, false, false, false, false,  // 5-9
        false, false, false, false, false,  // 10-14
        false, false, false, false, false,  // 15-19
    };
    if (coverageIdx < 0 || coverageIdx >= 20) {
        return true;
    }
    return kMandatory[coverageIdx];
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
    const float sep = 2.62f * gx;
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

/**
 * Undirected pairs from ``Travelers/docs/data/YQH26_data/instance_incompatible_pairs.csv``
 * (same rows ``insurance_model.py`` loads into ``compatibility_rules`` with ``compatible=False``).
 * Bundling page example: gap_insurance ✗ personal_property_floater; all listed pairs are optional add-ons.
 */
struct CoverageIncompatiblePair {
    int a;
    int b;
};

static constexpr CoverageIncompatiblePair kCoverageIncompatiblePairs[] = {
    {13, 14},  // gap_insurance, personal_property_floater
    {15, 19},  // additional_living_expense, earthquake_insurance
    {17, 18},  // excess_liability, flood_insurance
};

/** Red broken segment–gap–segment with an X in the gap (between marble surfaces). */
static void DrawIncompatibleBrokenLineBetweenMarbles(Vector3 centerFrom, Vector3 centerTo, float marbleR, Color c) {
    Vector3 d = Vector3Subtract(centerTo, centerFrom);
    const float len = Vector3Length(d);
    const float inset = marbleR * 1.08f;
    if (len < 2.f * inset + 1e-4f) {
        return;
    }
    Vector3 dir = Vector3Scale(d, 1.f / len);
    const Vector3 a = Vector3Add(centerFrom, Vector3Scale(dir, inset));
    const Vector3 b = Vector3Subtract(centerTo, Vector3Scale(dir, inset));
    const Vector3 ab = Vector3Subtract(b, a);
    const float L = Vector3Length(ab);
    if (L < 1e-5f) {
        return;
    }
    const Vector3 u = Vector3Scale(ab, 1.f / L);

    float halfGap = std::max(marbleR * 0.55f, L * 0.11f);
    halfGap = std::min(halfGap, L * 0.42f);

    const float tMid = L * 0.5f;
    const float tEnd1 = tMid - halfGap;
    const float tStart2 = tMid + halfGap;

    const Vector3 pGapA = Vector3Add(a, Vector3Scale(u, tEnd1));
    const Vector3 pGapB = Vector3Add(a, Vector3Scale(u, tStart2));
    DrawLine3D(a, pGapA, c);
    DrawLine3D(pGapB, b, c);

    const Vector3 mid = Vector3Add(a, Vector3Scale(u, tMid));

    Vector3 up = {0.f, 1.f, 0.f};
    if (std::fabs(Vector3DotProduct(u, up)) > 0.92f) {
        up = {1.f, 0.f, 0.f};
    }
    Vector3 v = Vector3Normalize(Vector3CrossProduct(up, u));
    Vector3 w = Vector3Normalize(Vector3CrossProduct(u, v));
    const float xArm = std::min(halfGap * 0.65f, marbleR * 0.95f);
    const float k = 0.70710678f * xArm;
    const Vector3 vw = Vector3Add(v, w);
    const Vector3 vmw = Vector3Subtract(v, w);
    const float xOffset = std::min(halfGap * 0.38f, marbleR * 0.55f);
    const Vector3 midL = Vector3Subtract(mid, Vector3Scale(u, xOffset));
    const Vector3 midR = Vector3Add(mid, Vector3Scale(u, xOffset));
    auto drawX = [&](Vector3 c0) {
        DrawLine3D(Vector3Subtract(c0, Vector3Scale(vw, k)), Vector3Add(c0, Vector3Scale(vw, k)), c);
        DrawLine3D(Vector3Subtract(c0, Vector3Scale(vmw, k)), Vector3Add(c0, Vector3Scale(vmw, k)), c);
    };
    drawX(midL);
    drawX(midR);
}

// --- Bundler story (``01_insurance_bundling.html`` pkg "Suburban Homeowner": deps + incompat in play) ---
enum class BundlerStoryPhase : int {
    FadeScene = 0,
    Checkmarks,
    RiseStaging,
    SuccessFlood,
    FailFloaterDep,
    FailExcessIncompat,
    WrapBundle,
    Done
};

/** ``PACKAGES[2]`` coverages: enhanced auto, homeowners, collision, UM, gap, umbrella, flood. */
static constexpr int kDemoPkgCov[] = {1, 2, 5, 9, 13, 16, 18};
static constexpr int kDemoStagingCov[] = {1, 2, 5, 9, 13, 16};  // package minus flood (star beat) */
static constexpr int kDemoSuccessCov = 18;   // flood — requires homeowners (2), in package */
static constexpr int kDemoSuccessReq = 2;
static constexpr int kDemoFailDepCov = 14;   // floater — requires condo (3), not in this package */
static constexpr int kDemoFailDepReq = 3;
static constexpr int kDemoFailIncompatCov = 17;  // excess — needs enhanced (1) OK but ✗ flood (18) in bundle */
static constexpr int kDemoIncompatPartner = 18;

static bool DemoPackageContains(int covIdx) {
    for (int c : kDemoPkgCov) {
        if (c == covIdx) {
            return true;
        }
    }
    return false;
}

static bool DemoIsStagingMarble(int covIdx) {
    for (int c : kDemoStagingCov) {
        if (c == covIdx) {
            return true;
        }
    }
    return false;
}

static void BundlerDecodePhase(float tSec, BundlerStoryPhase* outPh, float* outLocal) {
    static const float kDur[] = {0.72f, 0.58f, 1.05f, 2.7f, 2.75f, 2.75f, 2.05f};
    const int n = static_cast<int>(sizeof(kDur) / sizeof(kDur[0]));
    float acc = 0.f;
    for (int i = 0; i < n; i++) {
        if (tSec < acc + kDur[i]) {
            *outPh = static_cast<BundlerStoryPhase>(i);
            *outLocal = (tSec - acc) / kDur[i];
            return;
        }
        acc += kDur[i];
    }
    *outPh = BundlerStoryPhase::Done;
    *outLocal = 1.f;
}

static Vector3 BundlerTrayCenter(float ox, float oz, float gx, float gz, int n, float landY, float amp) {
    const float zn = static_cast<float>(std::max(0, n - 1));
    return {ox + gx * 1.45f, landY + amp * 0.5f, oz + gz * zn * 0.9f};
}

static void BundlerBaseXZ(int covIdx, int n, int nMarbles, float ox, float oz, float gx, float gz, float* lx,
                          float* lz) {
    float cx = 0.f;
    float cz = 0.f;
    float sx = 0.f;
    float sz = 0.f;
    CoverageGroupLandXZ(covIdx, n, nMarbles, ox, oz, gx, gz, &cx, &cz);
    CoverageSpreadLandXZ(covIdx, n, nMarbles, ox, oz, gx, gz, &sx, &sz);
    const float spreadMulFinal = 1.68f;
    *lx = cx + (sx - cx) * spreadMulFinal;
    *lz = cz + (sz - cz) * spreadMulFinal;
}

static Vector3 StagingOffsetForSlot(int slot6, float radius, float riseY) {
    const float a = static_cast<float>(slot6) * (6.2831853f / 6.f) + 0.35f;
    return {std::cos(a) * radius, riseY, std::sin(a) * radius};
}

static int StagingSlotForCov(int covIdx) {
    for (int s = 0; s < 6; s++) {
        if (kDemoStagingCov[s] == covIdx) {
            return s;
        }
    }
    return 0;
}

static void DrawCheckmark3D(Vector3 tip, float s, Color col) {
    const Vector3 p0 = {tip.x - s * 0.42f, tip.y, tip.z - s * 0.18f};
    const Vector3 p1 = {tip.x - s * 0.08f, tip.y, tip.z + s * 0.32f};
    const Vector3 p2 = {tip.x + s * 0.52f, tip.y, tip.z - s * 0.42f};
    DrawLine3D(p0, p1, col);
    DrawLine3D(p1, p2, col);
}

static void DrawMiniRedX3D(Vector3 c, float arm, Color col) {
    const float k = 0.70710678f * arm;
    DrawLine3D({c.x - k, c.y, c.z - k}, {c.x + k, c.y, c.z + k}, col);
    DrawLine3D({c.x - k, c.y, c.z + k}, {c.x + k, c.y, c.z - k}, col);
}

static float BundlerFlashTwice(float u) {
    if (u < 0.18f) {
        return 1.f;
    }
    if (u < 0.36f) {
        return 0.1f;
    }
    if (u < 0.54f) {
        return 1.f;
    }
    if (u < 0.72f) {
        return 0.1f;
    }
    return 0.12f;
}

/** ILP objective split (``02_ilp_to_qubo.html``): $M_{i,m}=\\text{price}\\cdot\\text{margin}\\cdot(1-\\delta_m)$,
 *  $C_{i,m}$ full contribution including take rate, $\\alpha_{i,m}$, and $(1+\\beta\\delta_m)$.
 *  Values for **package index 2** (Suburban Homeowner, $\\delta=0.15$, $\\beta=1.2$) and AFFINITIES[2] from bundling page. */
static constexpr float kMcRowM[7] = {
    280.84f, 308.12f, 97.92f, 44.63f, 79.73f, 96.9f, 145.35f,
};
static constexpr float kMcRowC[7] = {
    178.95f, 307.24f, 71.64f, 25.28f, 18.07f, 28.59f, 25.22f,
};

/** Demo package coverage rows × duplicated columns (ILP M, C are full matrices; narrative uses row-constant tiles). */
static constexpr int kMcMatrixDim = 7;

enum class McStoryPhase : int {
    FadeNonPackage = 0,
    ShowMatrices,
    UnbundleWrap,
    GatherCenter,
    SplitHalves,
    AssignColumns,
    Done
};

static constexpr float kMcPhaseDur[6] = {0.52f, 0.7f, 0.58f, 1.05f, 0.72f, 1.38f};

static float McPhaseStartTime(McStoryPhase ph) {
    float acc = 0.f;
    for (int i = 0; i < static_cast<int>(ph); i++) {
        acc += kMcPhaseDur[i];
    }
    return acc;
}

static void McDecodePhase(float tSec, McStoryPhase* outPh, float* outLocal) {
    const int n = static_cast<int>(sizeof(kMcPhaseDur) / sizeof(kMcPhaseDur[0]));
    float acc = 0.f;
    for (int i = 0; i < n; i++) {
        if (tSec < acc + kMcPhaseDur[i]) {
            *outPh = static_cast<McStoryPhase>(i);
            *outLocal = (tSec - acc) / kMcPhaseDur[i];
            return;
        }
        acc += kMcPhaseDur[i];
    }
    *outPh = McStoryPhase::Done;
    *outLocal = 1.f;
}

static int DemoPkgRowIndex(int covIdx) {
    for (int i = 0; i < 7; i++) {
        if (kDemoPkgCov[i] == covIdx) {
            return i;
        }
    }
    return 0;
}

static void McMatrixGrids(float landY, float gx, float gz, float oz, int nGrid, float* outColStep, float* outRowStep,
                          Vector3 mOut[kMcMatrixDim][kMcMatrixDim], Vector3 cOut[kMcMatrixDim][kMcMatrixDim]) {
    const float colStep = gx * 0.44f;
    const float rowStep = gx * 1.02f;
    *outColStep = colStep;
    *outRowStep = rowStep;
    const float xCenterM = -gx * 3.35f;
    const float xCenterC = gx * 3.35f;
    const float zMid = oz + gz * static_cast<float>(std::max(0, nGrid - 1)) * 0.5f;
    for (int r = 0; r < kMcMatrixDim; r++) {
        const float z = zMid + rowStep * (3.f - static_cast<float>(r));
        for (int c = 0; c < kMcMatrixDim; c++) {
            const float xM = xCenterM + (static_cast<float>(c) - 3.f) * colStep;
            const float xC = xCenterC + (static_cast<float>(c) - 3.f) * colStep;
            mOut[r][c] = {xM, landY + 0.055f * gx, z};
            cOut[r][c] = {xC, landY + 0.055f * gx, z};
        }
    }
}

static void DrawMcMatrixPads(const Vector3 mGrid[kMcMatrixDim][kMcMatrixDim],
                             const Vector3 cGrid[kMcMatrixDim][kMcMatrixDim], float cellX, float cellZ,
                             const float* heatM, const float* heatC, const Color& coldC, const Color& hotC, float alpha) {
    auto heatTint = [](float v, float vmax, Color cold, Color hot, float a) -> Color {
        const float t = std::max(0.f, std::min(1.f, v / std::max(vmax, 1e-6f)));
        Color o = {
            static_cast<unsigned char>(std::round(static_cast<float>(cold.r) +
                                                 t * (static_cast<float>(hot.r) - static_cast<float>(cold.r)))),
            static_cast<unsigned char>(std::round(static_cast<float>(cold.g) +
                                                 t * (static_cast<float>(hot.g) - static_cast<float>(cold.g)))),
            static_cast<unsigned char>(std::round(static_cast<float>(cold.b) +
                                                 t * (static_cast<float>(hot.b) - static_cast<float>(cold.b)))),
            static_cast<unsigned char>(std::round(255.f * a))};
        return o;
    };
    float maxM = heatM[0];
    float maxC = heatC[0];
    for (int i = 1; i < kMcMatrixDim; i++) {
        maxM = std::max(maxM, heatM[i]);
        maxC = std::max(maxC, heatC[i]);
    }
    const Color coldM = {15, 40, 85, 255};
    const Color hotM = {0, 102, 204, 255};
    const float thickY = 0.035f * std::min(cellX, cellZ);
    const float hx = cellX * 0.44f;
    const float hz = cellZ * 0.44f;
    for (int r = 0; r < kMcMatrixDim; r++) {
        Color fm = heatTint(heatM[r], maxM, coldM, hotM, alpha * 0.55f);
        Color fc = heatTint(heatC[r], maxC, coldC, hotC, alpha * 0.55f);
        for (int c = 0; c < kMcMatrixDim; c++) {
            DrawCube(mGrid[r][c], hx * 2.f, thickY, hz * 2.f, fm);
            DrawCubeWires(mGrid[r][c], hx * 2.f, thickY, hz * 2.f,
                          {200, 210, 225, static_cast<unsigned char>(std::round(200.f * alpha))});
            DrawCube(cGrid[r][c], hx * 2.f, thickY, hz * 2.f, fc);
            DrawCubeWires(cGrid[r][c], hx * 2.f, thickY, hz * 2.f,
                          {200, 210, 225, static_cast<unsigned char>(std::round(180.f * alpha))});
        }
    }
}

static void DrawMcMatrixFrames(const Vector3 grid[kMcMatrixDim][kMcMatrixDim], float cellX, float cellZ, Color lineCol,
                               float alpha) {
    lineCol.a = static_cast<unsigned char>(std::round(static_cast<float>(lineCol.a) * alpha));
    const float hx = cellX * 0.48f;
    const float hz = cellZ * 0.48f;
    for (int r = 0; r < kMcMatrixDim; r++) {
        for (int c = 0; c < kMcMatrixDim; c++) {
            const Vector3 p = grid[r][c];
            const float x0 = p.x - hx;
            const float x1 = p.x + hx;
            const float z0 = p.z - hz;
            const float z1 = p.z + hz;
            const float y = p.y;
            DrawLine3D({x0, y, z0}, {x1, y, z0}, lineCol);
            DrawLine3D({x1, y, z0}, {x1, y, z1}, lineCol);
            DrawLine3D({x1, y, z1}, {x0, y, z1}, lineCol);
            DrawLine3D({x0, y, z1}, {x0, y, z0}, lineCol);
        }
    }
}

static void DrawMcMatrixCellBalls(const Vector3 grid[kMcMatrixDim][kMcMatrixDim], float cellX, float cellZ,
                                  const float* heatRow, const Color& coldTint, const Color& hotTint, Color wireCol,
                                  float gx, float alpha) {
    float vmax = heatRow[0];
    for (int i = 1; i < kMcMatrixDim; i++) {
        vmax = std::max(vmax, heatRow[i]);
    }
    auto tint = [&](float v) -> Color {
        const float t = std::max(0.f, std::min(1.f, v / std::max(vmax, 1e-6f)));
        Color o = {
            static_cast<unsigned char>(std::round(static_cast<float>(coldTint.r) +
                                                 t * (static_cast<float>(hotTint.r) - static_cast<float>(coldTint.r)))),
            static_cast<unsigned char>(std::round(static_cast<float>(coldTint.g) +
                                                 t * (static_cast<float>(hotTint.g) - static_cast<float>(coldTint.g)))),
            static_cast<unsigned char>(std::round(static_cast<float>(coldTint.b) +
                                                 t * (static_cast<float>(hotTint.b) - static_cast<float>(coldTint.b)))),
            static_cast<unsigned char>(std::round(255.f * alpha * 0.92f))};
        return o;
    };
    const float rad = std::min(cellX, cellZ) * 0.11f;
    const float lift = gx * 0.09f;
    wireCol.a = static_cast<unsigned char>(std::round(static_cast<float>(wireCol.a) * alpha));
    for (int r = 0; r < kMcMatrixDim; r++) {
        Color rowCol = tint(heatRow[r]);
        for (int c = 0; c < kMcMatrixDim; c++) {
            Vector3 p = grid[r][c];
            p.y += lift;
            DrawSphere(p, rad * 0.92f, rowCol);
            DrawSphereWires(p, rad, 7, 10, wireCol);
        }
    }
}

static Vector3 McNonPackagePlanePos(int covIdx, int nMarbles, int nGrid, float ox, float oz, float gx, float gz,
                                    float landY) {
    float lx = 0.f;
    float lz = 0.f;
    BundlerBaseXZ(covIdx, nGrid, nMarbles, ox, oz, gx, gz, &lx, &lz);
    return {lx, landY, lz};
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

static Vector3 McMatrixGridCenter(const Vector3 grid[kMcMatrixDim][kMcMatrixDim]) {
    float sx = 0.f;
    float sz = 0.f;
    for (int r = 0; r < kMcMatrixDim; r++) {
        for (int c = 0; c < kMcMatrixDim; c++) {
            sx += grid[r][c].x;
            sz += grid[r][c].z;
        }
    }
    const float inv = 1.f / static_cast<float>(kMcMatrixDim * kMcMatrixDim);
    const float y = grid[0][0].y;
    return {sx * inv, y, sz * inv};
}

/** Large "M" / "C" HUD: slower double-flash when matrices appear, then stay fully lit until the story advances (→). */
static constexpr float kMcMatrixLabelFlashSec = 1.05f;

static float McMatrixLabelAlpha(float mcElapsedSec) {
    const float t0 = McPhaseStartTime(McStoryPhase::ShowMatrices);
    if (mcElapsedSec < t0) {
        return 0.f;
    }
    const float u = mcElapsedSec - t0;
    if (u < kMcMatrixLabelFlashSec) {
        return BundlerFlashTwice(std::min(1.f, u / kMcMatrixLabelFlashSec));
    }
    return 1.f;
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

enum class McMergeStoryPhase : int {
    FadeOverlays = 0,
    LiftCSeven,
    ExpandFullStack,
    MergeTeal,
    Done
};

static constexpr float kMcMergePhaseDur[4] = {0.55f, 0.72f, 1.9f, 1.5f};

static void McMergeDecodePhase(float tSec, McMergeStoryPhase* outPh, float* outLocal) {
    float acc = 0.f;
    for (int i = 0; i < 4; i++) {
        if (tSec < acc + kMcMergePhaseDur[i]) {
            *outPh = static_cast<McMergeStoryPhase>(i);
            *outLocal = (tSec - acc) / kMcMergePhaseDur[i];
            return;
        }
        acc += kMcMergePhaseDur[i];
    }
    *outPh = McMergeStoryPhase::Done;
    *outLocal = 1.f;
}

/** Fake block-diagonal shading: off-block regions (between diagonal Q blocks) read darker — illustrative zeros. */
static Color QBlockDiagonalTealCell(Color baseFill, const Color& baseWire, int row, int col, int blockSide,
                                    Color* outWire) {
    if (blockSide <= 0) {
        *outWire = baseWire;
        return baseFill;
    }
    const int br = row / blockSide;
    const int bc = col / blockSide;
    if (br == bc) {
        *outWire = baseWire;
        return baseFill;
    }
    const Color dimFill = {0, 36, 46, 255};
    Color fc = LerpColorRgb(baseFill, dimFill, 0.76f);
    fc.a = baseFill.a;
    const Color dimWire = {25, 65, 72, baseWire.a};
    *outWire = LerpColorRgb(baseWire, dimWire, 0.62f);
    outWire->a = baseWire.a;
    return fc;
}

/** Procedural N×M tiles (coverage × package) for the full objective-coefficient view. */
static void DrawInsuranceCoefficientMatrix(int nRows, int nCols, float cx, float cz, float yPlane, float colW,
                                           float rowW, float thickY, const Color& cold, const Color& hot, Color wireCol,
                                           float alpha, int blockDiagSide) {
    wireCol.a = static_cast<unsigned char>(
        std::round(static_cast<float>(wireCol.a) * std::max(0.f, std::min(1.f, alpha))));
    for (int r = 0; r < nRows; r++) {
        for (int c = 0; c < nCols; c++) {
            const float shade = 0.32f + 0.68f * (0.5f + 0.5f * sinf(r * 0.21f + c * 0.13f));
            Color fc = LerpColorRgb(cold, hot, shade);
            fc.a = static_cast<unsigned char>(std::round(255.f * std::max(0.f, std::min(1.f, alpha)) * 0.52f));
            Color wcell = wireCol;
            fc = QBlockDiagonalTealCell(fc, wireCol, r, c, blockDiagSide, &wcell);
            const float x = cx - static_cast<float>(nCols - 1) * 0.5f * colW + static_cast<float>(c) * colW;
            const float z = cz + static_cast<float>(nRows - 1) * 0.5f * rowW - static_cast<float>(r) * rowW;
            const Vector3 p = {x, yPlane, z};
            DrawCube(p, colW * 0.9f, thickY, rowW * 0.9f, fc);
            DrawCubeWires(p, colW * 0.9f, thickY, rowW * 0.9f, wcell);
        }
    }
}

static void McCopyGridLiftC(const Vector3 src[kMcMatrixDim][kMcMatrixDim], float dy,
                            Vector3 dst[kMcMatrixDim][kMcMatrixDim]) {
    for (int r = 0; r < kMcMatrixDim; r++) {
        for (int c = 0; c < kMcMatrixDim; c++) {
            dst[r][c] = src[r][c];
            dst[r][c].y += dy;
        }
    }
}

enum class QSliceStoryPhase : int {
    ShowGrid = 0,
    HighlightSel,
    FadeOthers,
    ExpandMergeMesh,
    Fade2D,
    Done
};

static constexpr float kQSlicePhaseDur[5] = {0.5f, 0.55f, 0.75f, 1.55f, 0.65f};

static void QSliceDecodePhase(float tSec, QSliceStoryPhase* outPh, float* outLocal) {
    float acc = 0.f;
    for (int i = 0; i < 5; i++) {
        if (tSec < acc + kQSlicePhaseDur[i]) {
            *outPh = static_cast<QSliceStoryPhase>(i);
            *outLocal = (tSec - acc) / kQSlicePhaseDur[i];
            return;
        }
        acc += kQSlicePhaseDur[i];
    }
    *outPh = QSliceStoryPhase::Done;
    *outLocal = 1.f;
}

/** Full n×n teal Q plane with tile masking, optional morph of the selected qMeshN×qMeshN block toward mesh XZ layout. */
static void DrawTealQMatrixDynamic(int n, float cx, float cz, float yPlane, float colW, float rowW, float thickY,
                                   const Color& cold, const Color& hot, Color wireCol, float baseAlpha, int tileSide,
                                   int selTI, int selTJ, float alphaOther, float alphaSelected, float highlightBoost,
                                   int blockBaseI, int blockBaseJ, int qMeshN, float morphU, float oxMesh, float ozMesh,
                                   float meshStepX, float meshStepZ) {
    const int ts = std::max(1, std::min(tileSide, n));
    wireCol.a = static_cast<unsigned char>(
        std::round(static_cast<float>(wireCol.a) * std::max(0.f, std::min(1.f, baseAlpha))));
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            const int ti = r / ts;
            const int tj = c / ts;
            const bool inSel =
                (r >= blockBaseI && r < blockBaseI + qMeshN && c >= blockBaseJ && c < blockBaseJ + qMeshN);
            const bool isSelTile = (ti == selTI && tj == selTJ);
            float cellA = baseAlpha * (isSelTile ? alphaSelected : alphaOther);
            if (isSelTile && highlightBoost > 0.f) {
                cellA = std::min(1.f, cellA * (1.f + highlightBoost));
            }
            if (cellA < 0.004f) {
                continue;
            }
            Vector3 p0 = {cx - static_cast<float>(n - 1) * 0.5f * colW + static_cast<float>(c) * colW, yPlane,
                          cz + static_cast<float>(n - 1) * 0.5f * rowW - static_cast<float>(r) * rowW};
            Vector3 p = p0;
            float cw = colW * 0.9f;
            float rw = rowW * 0.9f;
            if (inSel && morphU > 0.001f) {
                const int li = r - blockBaseI;
                const int lj = c - blockBaseJ;
                const Vector3 p1 = {oxMesh + static_cast<float>(li) * meshStepX, yPlane,
                                    ozMesh + static_cast<float>(lj) * meshStepZ};
                const float e = EaseInOutCubic(std::max(0.f, std::min(1.f, morphU)));
                p = Vector3Lerp(p0, p1, e);
                // Match DrawQBlockMesh3D: morphed tiles use passed meshStepX/Z (full-plane span / (qMeshN-1)).
                cw = colW * 0.9f + (meshStepX - colW * 0.9f) * e;
                rw = rowW * 0.9f + (meshStepZ - rowW * 0.9f) * e;
            }
            const float shade = 0.32f + 0.68f * (0.5f + 0.5f * sinf(r * 0.11f + c * 0.09f));
            Color fc = LerpColorRgb(cold, hot, shade);
            fc.a = static_cast<unsigned char>(std::round(255.f * cellA * 0.52f));
            const int bdSide = std::max(1, std::min(qMeshN, n));
            Color wcell;
            fc = QBlockDiagonalTealCell(fc, wireCol, r, c, bdSide, &wcell);
            wcell.a = static_cast<unsigned char>(std::round(static_cast<float>(wcell.a) * cellA));
            const float th = thickY * (0.85f + 0.15f * (inSel ? EaseInOutCubic(std::max(0.f, std::min(1.f, morphU))) : 0.f));
            DrawCube(p, cw, th, rw, fc);
            DrawCubeWires(p, cw, th, rw, wcell);
        }
    }
}

static void DrawQBlockSliceLines(int n, float cx, float cz, float yPlane, float colW, float rowW, int tileSide,
                               Color lineCol, float alpha) {
    const int ts = std::max(1, std::min(tileSide, n));
    lineCol.a = static_cast<unsigned char>(
        std::round(static_cast<float>(lineCol.a) * std::max(0.f, std::min(1.f, alpha))));
    const float x0 = cx - static_cast<float>(n - 1) * 0.5f * colW;
    const float z0 = cz + static_cast<float>(n - 1) * 0.5f * rowW;
    const float x1 = x0 + static_cast<float>(n - 1) * colW;
    const float z1 = z0 - static_cast<float>(n - 1) * rowW;
    for (int k = ts; k < n; k += ts) {
        const float x = x0 + static_cast<float>(k) * colW;
        DrawLine3D({x, yPlane, z0}, {x, yPlane, z1}, lineCol);
        const float z = z0 - static_cast<float>(k) * rowW;
        DrawLine3D({x0, yPlane, z}, {x1, yPlane, z}, lineCol);
    }
}

/** Same height rule as the Q-block wire mesh (raw Q vs Hamiltonian blend, positive-peak gain). */
static float QBlockMeshVertexHeight(const QuboSurface& surf, int gi, int gj, double qij, double scaleDen,
                                    float ampVis, float valueExtrude, float coeffToEnergyBlend,
                                    bool energyHeightAlternatingDemo) {
    const float ve = std::max(0.f, std::min(1.f, valueExtrude));
    const float blend = std::max(0.f, std::min(1.f, coeffToEnergyBlend));
    const bool haveX = SurfHasAssignment(surf);
    const double hCoeff = qij;
    double xi = 0.0;
    double xj = 0.0;
    if (energyHeightAlternatingDemo) {
        xi = static_cast<double>(gi & 1);
        xj = static_cast<double>(gj & 1);
    } else if (haveX && gi >= 0 && gj >= 0 && gi < surf.n && gj < surf.n) {
        xi = static_cast<double>(surf.x[static_cast<size_t>(gi)]);
        xj = static_cast<double>(surf.x[static_cast<size_t>(gj)]);
    } else {
        xi = static_cast<double>(gi & 1);
        xj = static_cast<double>(gj & 1);
    }
    const double hEnergy = qij * xi * xj;
    const double h = hCoeff * (1.0 - static_cast<double>(blend)) + hEnergy * static_cast<double>(blend);
    float hn = static_cast<float>(h / scaleDen);
    if (hn > 0.f) {
        hn *= kQBlockPositivePeakGain;
    }
    return hn * ampVis * ve;
}

static float SampleQBlockMeshHeightBilinear(const QuboSurface& surf, int bi, int bj, int qn, float u, float v,
                                            double scaleDen, float ampVis, float valueExtrude,
                                            float coeffToEnergyBlend, bool energyHeightAlternatingDemo) {
    if (qn <= 0) {
        return 0.f;
    }
    if (qn == 1) {
        const int gi = bi;
        const int gj = bj;
        return QBlockMeshVertexHeight(surf, gi, gj, surf.qAt(gi, gj), scaleDen, ampVis, valueExtrude,
                                      coeffToEnergyBlend, energyHeightAlternatingDemo);
    }
    u = std::max(0.f, std::min(static_cast<float>(qn - 1), u));
    v = std::max(0.f, std::min(static_cast<float>(qn - 1), v));
    const int i0 = static_cast<int>(std::floor(u));
    const int j0 = static_cast<int>(std::floor(v));
    const int i1 = std::min(i0 + 1, qn - 1);
    const int j1 = std::min(j0 + 1, qn - 1);
    const float fu = u - static_cast<float>(i0);
    const float fv = v - static_cast<float>(j0);
    auto corner = [&](int ii, int jj) {
        const int gi = bi + ii;
        const int gj = bj + jj;
        return QBlockMeshVertexHeight(surf, gi, gj, surf.qAt(gi, gj), scaleDen, ampVis, valueExtrude,
                                      coeffToEnergyBlend, energyHeightAlternatingDemo);
    };
    const float h00 = corner(i0, j0);
    const float h10 = corner(i1, j0);
    const float h01 = corner(i0, j1);
    const float h11 = corner(i1, j1);
    const float h0 = h00 * (1.f - fu) + h10 * fu;
    const float h1 = h01 * (1.f - fu) + h11 * fu;
    return h0 * (1.f - fv) + h1 * fv;
}

static void DrawQBlockMesh3D(const QuboSurface& surf, int bi, int bj, int qn, float oxB, float ozB, float gx, float gz,
                             double scaleDen, float ampVis, float meshAlpha, float diagAlpha, float assignAlpha,
                             float valueExtrude = 1.f, float coeffToEnergyBlend = 0.f,
                             bool energyHeightAlternatingDemo = false) {
    if (qn <= 0) {
        return;
    }
    const float ve = std::max(0.f, std::min(1.f, valueExtrude));
    const float blend = std::max(0.f, std::min(1.f, coeffToEnergyBlend));
    Color meshLine = g_style.meshLine;
    if (blend > 0.004f) {
        const Color energyTint = {175, 210, 255, meshLine.a};
        meshLine = LerpColorRgb(meshLine, energyTint, blend * 0.78f);
    }
    meshLine.a = static_cast<unsigned char>(std::round(static_cast<float>(meshLine.a) * meshAlpha));
    Color diagLine = g_style.diagonalLine;
    diagLine.a = static_cast<unsigned char>(std::round(static_cast<float>(diagLine.a) * diagAlpha));
    Color assignLine = g_style.assignmentLine;
    assignLine.a = static_cast<unsigned char>(std::round(static_cast<float>(assignLine.a) * assignAlpha));

    for (int i = 0; i < qn; i++) {
        for (int j = 0; j < qn; j++) {
            const int gi = bi + i;
            const int gj = bj + j;
            double qij = surf.qAt(gi, gj);
            float y = QBlockMeshVertexHeight(surf, gi, gj, qij, scaleDen, ampVis, valueExtrude, coeffToEnergyBlend,
                                             energyHeightAlternatingDemo);
            Vector3 p = {oxB + static_cast<float>(i) * gx, y, ozB + static_cast<float>(j) * gz};
            if (i + 1 < qn) {
                const int gi2 = bi + i + 1;
                double qnxt = surf.qAt(gi2, gj);
                float yn = QBlockMeshVertexHeight(surf, gi2, gj, qnxt, scaleDen, ampVis, valueExtrude,
                                                  coeffToEnergyBlend, energyHeightAlternatingDemo);
                DrawLine3D(p, {oxB + static_cast<float>(i + 1) * gx, yn, ozB + static_cast<float>(j) * gz}, meshLine);
            }
            if (j + 1 < qn) {
                const int gj2 = bj + j + 1;
                double qnxt = surf.qAt(gi, gj2);
                float yn = QBlockMeshVertexHeight(surf, gi, gj2, qnxt, scaleDen, ampVis, valueExtrude,
                                                  coeffToEnergyBlend, energyHeightAlternatingDemo);
                DrawLine3D(p, {oxB + static_cast<float>(i) * gx, yn, ozB + static_cast<float>(j + 1) * gz}, meshLine);
            }
            if (gi == gj && ve > 0.004f) {
                DrawLine3D({p.x, 0.f, p.z}, p, diagLine);
            }
        }
    }

    if (!surf.x.empty() && ve > 0.02f) {
        for (int i = 0; i < qn; i++) {
            for (int j = 0; j < qn; j++) {
                const int gi = bi + i;
                const int gj = bj + j;
                if (gi != gj || gi < 0 || gi >= surf.n) {
                    continue;
                }
                if (!surf.x[static_cast<size_t>(gi)]) {
                    continue;
                }
                double qii = surf.qAt(gi, gj);
                float y = QBlockMeshVertexHeight(surf, gi, gj, qii, scaleDen, ampVis, valueExtrude, coeffToEnergyBlend,
                                                 energyHeightAlternatingDemo);
                Vector3 p = {oxB + static_cast<float>(i) * gx, y, ozB + static_cast<float>(j) * gz};
                float top = p.y + 0.35f * ampVis * ve;
                DrawLine3D(p, {p.x, top, p.z}, assignLine);
                DrawSphere({p.x, top, p.z}, 0.08f * ampVis * ve, assignLine);
            }
        }
    }
}

struct HamRollingBall {
    float u = 0.f;
    float v = 0.f;
    float vu = 0.f;
    float vv = 0.f;
    Color wire{};
    Color fill{};
};

static Color HamRollingBallWire(int idx) {
    static const Color palette[] = {
        {0, 102, 204, 255},
        {0, 53, 107, 255},
        {45, 140, 60, 255},
        {120, 200, 255, 255},
    };
    return palette[idx & 3];
}

static void InitHamRollingBalls(std::vector<HamRollingBall>& balls, int qn, uint32_t salty) {
    constexpr int kN = 14;
    if (qn < 2) {
        balls.clear();
        return;
    }
    balls.resize(kN);
    const float span = static_cast<float>(qn - 1);
    for (int i = 0; i < kN; i++) {
        const float g = std::fmod(0.6180339887f * static_cast<float>(i + 1) + static_cast<float>(salty & 4095u) * 0.001f,
                                  1.f);
        const float h = std::fmod(0.4142135623f * static_cast<float>(i + 7) + static_cast<float>(salty >> 12) * 0.0023f,
                                  1.f);
        HamRollingBall b;
        b.u = 0.11f * span + g * 0.78f * span;
        b.v = 0.11f * span + h * 0.78f * span;
        b.vu = 0.f;
        b.vv = 0.f;
        b.wire = HamRollingBallWire(i);
        b.fill = b.wire;
        b.fill.a = 210;
        balls[static_cast<size_t>(i)] = b;
    }
}

static void ClampRolling1D(float& pos, float& vel, float lo, float hi) {
    if (pos < lo) {
        pos = lo;
        if (vel < 0.f) {
            vel = -vel * 0.38f;
        }
    }
    if (pos > hi) {
        pos = hi;
        if (vel > 0.f) {
            vel = -vel * 0.38f;
        }
    }
}

static void UpdateHamRollingBalls(std::vector<HamRollingBall>& balls, const QuboSurface& surf, int bi, int bj, int qn,
                                  double scaleDen, float ampVis, float valueExtrude, float blend, bool energyAlt,
                                  float dt) {
    if (qn < 2 || balls.empty() || dt <= 0.f) {
        return;
    }
    const float span = static_cast<float>(qn - 1);
    float eps = 0.07f * span;
    if (eps < 0.045f) {
        eps = 0.045f;
    }
    if (eps > 0.24f) {
        eps = 0.24f;
    }
    const float accelK = 32.f;
    const float friction = std::exp(-6.5f * dt);
    for (auto& b : balls) {
        auto hSample = [&](float su, float sv) {
            return SampleQBlockMeshHeightBilinear(surf, bi, bj, qn, su, sv, scaleDen, ampVis, valueExtrude, blend,
                                                  energyAlt);
        };
        const float u = b.u;
        const float v = b.v;
        const float um = std::max(0.f, u - eps);
        const float up = std::min(span, u + eps);
        const float vm = std::max(0.f, v - eps);
        const float vp = std::min(span, v + eps);
        const float duEff = up - um;
        const float dvEff = vp - vm;
        const float dhu = (duEff > 1e-5f) ? (hSample(up, v) - hSample(um, v)) / duEff : 0.f;
        const float dhv = (dvEff > 1e-5f) ? (hSample(u, vp) - hSample(u, vm)) / dvEff : 0.f;
        b.vu += -accelK * dhu * dt;
        b.vv += -accelK * dhv * dt;
        b.vu *= friction;
        b.vv *= friction;
        const float v2 = b.vu * b.vu + b.vv * b.vv;
        const float g2 = dhu * dhu + dhv * dhv;
        if (v2 < 0.00005f && g2 < 1e-10f) {
            b.vu = 0.f;
            b.vv = 0.f;
        } else if (v2 < 0.00028f && g2 < 6e-9f) {
            b.vu *= 0.82f;
            b.vv *= 0.82f;
        }
        b.u += b.vu * dt;
        b.v += b.vv * dt;
        ClampRolling1D(b.u, b.vu, 0.f, span);
        ClampRolling1D(b.v, b.vv, 0.f, span);
    }
}

enum class QaoaStoryPhase : int {
    PickDiagonal = 0,
    PickOffDiagonal,
    RiseUp,
    GridFade,
    MorphGates,
    DashToQaoa,
    BoxSpinSpit,
    HoldOutro,
    Count
};

static void QaoaDecodePhase(float tSec, QaoaStoryPhase* outPh, float* outLocal) {
    const float a0 = 0.82f;
    const float a1 = a0 + 0.82f;
    const float a2 = a1 + 1.22f;
    const float a3 = a2 + 0.78f;
    const float a4 = a3 + 0.68f;
    const float a5 = a4 + 1.05f;
    const float a6 = a5 + 1.45f;
    if (tSec < 0.f) {
        *outPh = QaoaStoryPhase::PickDiagonal;
        *outLocal = 0.f;
        return;
    }
    if (tSec < a0) {
        *outPh = QaoaStoryPhase::PickDiagonal;
        *outLocal = tSec / a0;
        return;
    }
    if (tSec < a1) {
        *outPh = QaoaStoryPhase::PickOffDiagonal;
        *outLocal = (tSec - a0) / (a1 - a0);
        return;
    }
    if (tSec < a2) {
        *outPh = QaoaStoryPhase::RiseUp;
        *outLocal = (tSec - a1) / (a2 - a1);
        return;
    }
    if (tSec < a3) {
        *outPh = QaoaStoryPhase::GridFade;
        *outLocal = (tSec - a2) / (a3 - a2);
        return;
    }
    if (tSec < a4) {
        *outPh = QaoaStoryPhase::MorphGates;
        *outLocal = (tSec - a3) / (a4 - a3);
        return;
    }
    if (tSec < a5) {
        *outPh = QaoaStoryPhase::DashToQaoa;
        *outLocal = (tSec - a4) / (a5 - a4);
        return;
    }
    if (tSec < a6) {
        *outPh = QaoaStoryPhase::BoxSpinSpit;
        *outLocal = (tSec - a5) / (a6 - a5);
        return;
    }
    *outPh = QaoaStoryPhase::HoldOutro;
    *outLocal = 1.f;
}

static void FindQaoaLowestVertices(const QuboSurface& surf, int bi, int bj, int qn, double scaleDen, float ampVis,
                                   bool energyAlt, int* outDiagI, int* outOffI, int* outOffJ) {
    *outDiagI = 0;
    *outOffI = 0;
    *outOffJ = (qn > 1) ? 1 : 0;
    if (qn <= 0) {
        return;
    }
    float bestD = 1e30f;
    for (int i = 0; i < qn; i++) {
        const int gi = bi + i;
        const int gj = bj + i;
        const float h =
            QBlockMeshVertexHeight(surf, gi, gj, surf.qAt(gi, gj), scaleDen, ampVis, 1.f, 1.f, energyAlt);
        if (h < bestD) {
            bestD = h;
            *outDiagI = i;
        }
    }
    if (qn < 2) {
        return;
    }
    float bestO = 1e30f;
    bool any = false;
    for (int i = 0; i < qn; i++) {
        for (int j = 0; j < qn; j++) {
            if (i == j) {
                continue;
            }
            const int gi = bi + i;
            const int gj = bj + j;
            const float h =
                QBlockMeshVertexHeight(surf, gi, gj, surf.qAt(gi, gj), scaleDen, ampVis, 1.f, 1.f, energyAlt);
            if (h < bestO) {
                bestO = h;
                *outOffI = i;
                *outOffJ = j;
                any = true;
            }
        }
    }
    if (!any) {
        *outOffI = 0;
        *outOffJ = 1;
    }
}

static Vector3 QaoaBallWorld(const QuboSurface& surf, int bi, int bj, int qi, int qj, double scaleDen, float amp,
                             bool energyAlt, float oxB, float ozB, float gx, float gz, float marbleR,
                             float yLift) {
    const int gi = bi + qi;
    const int gj = bj + qj;
    const float y =
        QBlockMeshVertexHeight(surf, gi, gj, surf.qAt(gi, gj), scaleDen, amp, 1.f, 1.f, energyAlt);
    return {oxB + static_cast<float>(qi) * gx, y + marbleR + yLift, ozB + static_cast<float>(qj) * gz};
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

static void DrawQaoaBestBundlesMatrix2D(const QuboSurface& surf, int bi, int bj, int qn, int sw, int sh,
                                        double maxAbsQ, float alpha, int risePx = 0) {
    (void)sh;
    if (alpha < 0.02f) {
        return;
    }
    const int fs = 28;
    const char* title = "best bundles";
    const int tw = MeasureText(title, fs);
    const int cx = sw / 2;
    const int ty = 96 - risePx;
    Color tc = {220, 230, 245, static_cast<unsigned char>(std::min(255.f, alpha * 255.f))};
    DrawText(title, cx - tw / 2, ty, fs, tc);
    const int cell = std::max(12, std::min(32, 420 / std::max(1, qn)));
    const int gridW = qn * cell;
    const int gx0 = cx - gridW / 2;
    const int gy0 = ty + fs + 18;
    for (int i = 0; i < qn; i++) {
        for (int j = 0; j < qn; j++) {
            const double v = surf.qAt(bi + i, bj + j);
            Color c = QHeatColor(v, maxAbsQ);
            c.a = static_cast<unsigned char>(std::round(static_cast<float>(c.a) * alpha));
            DrawRectangle(gx0 + j * cell, gy0 + i * cell, cell - 1, cell - 1, c);
        }
    }
    Color frame = {0, 102, 204, static_cast<unsigned char>(std::min(255.f, alpha * 220.f))};
    DrawRectangleLines(gx0 - 3, gy0 - 3, gridW + 6, qn * cell + 6, frame);
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

static void DrawLegendPanel(int sw, int sh, const QuboSurface& surf, bool showAssignmentKey,
                            bool showParametricParams, double lambdaLive, double marginScaleS,
                            bool marginScaleEnabled) {
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

    if (showParametricParams) {
        DrawText("Interactive Q (v2 surface file)", px, y, fs + 2, title);
        y += lh + 4;
        DrawText("These sliders rebuild Q from exported pieces; penalties stay consistent.", px, y, fs, body);
        y += lh + 6;
        char lamL[280];
        std::snprintf(lamL, sizeof(lamL),
                      "  lambda (penalty weight):  %.6g   — scales ALL constraint squares the same.", lambdaLive);
        DrawText(lamL, px, y, fs, accent);
        y += lh;
        DrawText("  Too small: feasible assignments can lose to high-margin violations.", px, y, fs, body);
        y += lh;
        DrawText("  Too large: landscape dominated by feasibility; margins harder to read.", px, y, fs, body);
        y += lh;
        DrawText("  Drag the cyan/blue bottom slider (log scale).", px, y, fs, body);
        y += lh + 6;
        char sL[320];
        std::snprintf(sL, sizeof(sL),
                      "  s (margin scale):  %s   value %.6g",
                      marginScaleEnabled ? "ON " : "OFF (press M to enable)",
                      marginScaleEnabled ? marginScaleS : 1.0);
        DrawText(sL, px, y, fs, (Color){100, 170, 230, 255});
        y += lh;
        DrawText("  Scales only the profit/margin diagonal (coverage coefficients), not penalties.", px, y, fs, body);
        y += lh;
        DrawText("  Use it to see relative strength of margin vs lambda-weighted constraints.", px, y, fs, body);
        y += lh;
        DrawText("  When ON, drag the Yale-blue slider above lambda (log scale).", px, y, fs, body);
        y += lh + 8;
    }

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
    MarblesDependencies = 2,  // spread + dependency arrows (instance_dependencies.csv)
    MarblesIncompatibilities = 3,  // same layout + broken red incompatibility links (instance_incompatible_pairs.csv)
    MarblesBundler = 4,  // bundler tray demo (Suburban Homeowner package)
    MarblesMcDecompose = 5,  // M vs C column split (ILP objective factors, pkg 2)
    MarblesMcMerge = 6,      // full n×n Q (coverages + slacks) → stack → single teal matrix
    QSliceToBlock = 7,       // slice n×n → pick block → morph into smaller Q mesh
    QBlockField = 8,
    /** Smooth blend from raw Q_ij heights to pairwise term heights Q_ij x_i x_j (same grid; Hamiltonian / energy view). */
    HamiltonianLandscape = 9,
    /** Lowest diagonal / off-diagonal sites → Z / ZZ → QAOA box → “best bundles” + block Q grid (press → from Hamiltonian). */
    QaoaGateDemo = 10,
    Count = 11,
};

static constexpr float kFamilyColorBlendSec = 0.85f;
static constexpr float kFamilyPauseAfterColorSec = 0.5f;
static constexpr float kFamilyGroupMoveSec = 1.15f;
static constexpr float kDepSpreadSec = 0.95f;
static constexpr float kIncompatDrawSec = 0.9f;

static Font gStoryEqFont{};
static bool gStoryEqFontLoaded = false;

/** UTF-8 corpus covering every glyph used in on-screen equations (Noto Sans). */
static const char kEqFontGlyphCorpus[] =
    u8"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    u8"0123456789"
    u8" ,.;:_+-*/()[]{}^'=<>"
    u8"QEMCHPqijkmpstvAGSuvhZUJBX"
    u8"\u2208\u2264\u2265\u2212\u00b7\u00d7\u03b4\u03bb\u03b3\u03b2\u2207\u2211\u220f\u2190\u2202\u221d";

static bool InitStoryEquationFont() {
    const char* paths[] = {
        "fonts/NotoSans-Regular.ttf",
        "subprojects/will/visualizations/qubo_vis/fonts/NotoSans-Regular.ttf",
    };
    const char* chosen = nullptr;
    for (const char* p : paths) {
        if (FileExists(p)) {
            chosen = p;
            break;
        }
    }
    if (!chosen) {
        return false;
    }
    int cpCount = 0;
    int* cps = LoadCodepoints(kEqFontGlyphCorpus, &cpCount);
    if (!cps || cpCount <= 0) {
        return false;
    }
    gStoryEqFont = LoadFontEx(chosen, 96, cps, cpCount);
    UnloadCodepoints(cps);
    if (gStoryEqFont.glyphCount <= 0) {
        return false;
    }
    SetTextureFilter(gStoryEqFont.texture, TEXTURE_FILTER_TRILINEAR);
    gStoryEqFontLoaded = true;
    return true;
}

static void ShutdownStoryEquationFont() {
    if (gStoryEqFontLoaded) {
        UnloadFont(gStoryEqFont);
        gStoryEqFont = {};
        gStoryEqFontLoaded = false;
    }
}

/** Equations only: large white, right-aligned (LaTeX-like Unicode; true LaTeX needs external render). */
static void DrawStoryMathPanel(StoryStep step, int sw, int sh, bool showLegend, bool showQMatrix,
                               const QMatrixLayout& qLay, const QuboSurface& surf) {
    const float fontSize = gStoryEqFontLoaded ? 34.f : 22.f;
    const float spacing = 2.f;
    const float lineStep = fontSize + 10.f;
    const float marginR = 22.f;
    const int legReserve = showLegend ? (432 + 36) : 0;
    const float rightEdge = static_cast<float>(sw) - marginR - static_cast<float>(legReserve);

    float py = 56.f;
    if (showQMatrix) {
        const Rectangle qb = QMatrixHitBounds(qLay);
        const float belowQ = qb.y + qb.height + 16.f;
        if (belowQ + 120.f < static_cast<float>(sh) - 100.f) {
            py = std::max(py, belowQ);
        }
    }

    const Font font = gStoryEqFontLoaded ? gStoryEqFont : GetFontDefault();
    const Color white = {255, 255, 255, 255};

    auto drawEqLine = [&](const char* utf8Line) {
        Vector2 sz = MeasureTextEx(font, utf8Line, fontSize, spacing);
        float x = rightEdge - sz.x;
        if (x < 12.f) {
            x = 12.f;
        }
        DrawTextEx(font, utf8Line, {x, py}, fontSize, spacing, white);
        py += lineStep;
    };

    switch (step) {
    case StoryStep::MarblesDropWhite:
    case StoryStep::MarblesFamilyColor:
        drawEqLine(u8"x_i \u2208 {0,1}");
        break;
    case StoryStep::MarblesDependencies:
        drawEqLine(u8"x_j \u2264 x_i");
        break;
    case StoryStep::MarblesIncompatibilities:
        drawEqLine(u8"x_a + x_b \u2264 1");
        break;
    case StoryStep::MarblesBundler:
        drawEqLine(u8"x \u2208 {0,1}^K");
        break;
    case StoryStep::MarblesMcDecompose:
        drawEqLine(u8"M_{ij} = p_i m_i (1 \u2212 \u03b4_m)");
        drawEqLine(u8"C_{ij} = c_{i,m}");
        break;
    case StoryStep::MarblesMcMerge:
        drawEqLine(u8"E(x) = x^T Q x + c");
        break;
    case StoryStep::QSliceToBlock:
        drawEqLine(u8"P(t) = (1\u2212e(t))P_0 + e(t)P_1");
        if (surf.parametricLambda) {
            drawEqLine(u8"Q = s\u00b7diag(m) + \u03bb Q_{pen}");
        }
        break;
    case StoryStep::QBlockField:
        drawEqLine(u8"h_{ij} = (q_{ij}/S)\u00b7A\u00b7v");
        drawEqLine(u8"h_{ij} \u2190 G\u00b7h_{ij}   (h_{ij}>0)");
        if (surf.parametricLambda) {
            drawEqLine(u8"Q_{ij} = s\u00b7m_i\u03b4_{ij} + \u03bb Q^{pen}_{ij}");
            drawEqLine(u8"c = \u03bb\u00b7c_\u03bb");
        }
        break;
    case StoryStep::HamiltonianLandscape:
        drawEqLine(u8"H_{ij}(u)=(1\u2212u)Q_{ij}+u\u00b7Q_{ij}x_ix_j");
        drawEqLine(u8"h_{ij}=(H_{ij}/S_h)\u00b7A\u00b7v");
        drawEqLine(u8"h_{ij}\u2190 G\u00b7h_{ij}   (h_{ij}>0)");
        drawEqLine(u8"g \u221d \u2212\u2207 h");
        if (surf.parametricLambda) {
            drawEqLine(u8"Q_{ij} = s\u00b7m_i\u03b4_{ij} + \u03bb Q^{pen}_{ij}");
        }
        break;
    case StoryStep::QaoaGateDemo:
        drawEqLine(u8"H_C=\u2211_i h_i Z_i+\u2211_{i<j}J_{ij}Z_iZ_j");
        drawEqLine(u8"U_C(\u03b3)=e^{\u2212i\u03b3 H_C}");
        drawEqLine(u8"U_B(\u03b2)=\u220f_j e^{\u2212i\u03b2 X_j}");
        break;
    default:
        break;
    }
}

static float EaseOutCubic(float t) {
    if (t <= 0.f) return 0.f;
    if (t >= 1.f) return 1.f;
    const float u = 1.f - t;
    return 1.f - u * u * u;
}

static Vector3 BundlerStagingWorld(int covIdx, const Vector3& tray, float gx, float amp) {
    const Vector3 off = StagingOffsetForSlot(StagingSlotForCov(covIdx), gx * 2.55f, amp * 0.4f);
    return {tray.x + off.x, tray.y + off.y, tray.z + off.z};
}

static Vector3 BundlerPackageCluster(int covIdx, const Vector3& tray, float gx, float gz, float marbleR) {
    int ord = 0;
    bool found = false;
    for (int c : kDemoPkgCov) {
        if (c == covIdx) {
            found = true;
            break;
        }
        ord++;
    }
    if (!found) {
        return tray;
    }
    const float step = 6.2831853f / 8.f;
    const float a = static_cast<float>(ord) * step + 0.2f;
    const float rad = gx * 0.44f;
    return {tray.x + std::cos(a) * rad, tray.y + marbleR * 1.1f,
            tray.z + std::sin(a) * rad * (gz / std::max(gx, 0.001f))};
}

static void ComputeMcPackageMarble(int covIdx, int nMarbles, int nGrid, float /*ox*/, float oz, float gx, float gz,
                                   float landY, float amp, float marbleR, McStoryPhase ph, float u, const Vector3& tray,
                                   const Vector3 mGrid[kMcMatrixDim][kMcMatrixDim],
                                   const Vector3 cGrid[kMcMatrixDim][kMcMatrixDim], Vector3* outCenter, Vector3* outM,
                                   Vector3* outC, float* halfR, bool* drawMerged) {
    const Vector3 cluster = BundlerPackageCluster(covIdx, tray, gx, gz, marbleR);
    const int row = DemoPkgRowIndex(covIdx);
    const float zMid = oz + gz * static_cast<float>(std::max(0, nGrid - 1)) * 0.5f;
    const float gatherSpread = gx * 0.38f;
    const float ang = static_cast<float>(row) * (6.2831853f / 7.f);
    const Vector3 gather = {std::cos(ang) * gatherSpread * 0.35f, landY + amp * 0.58f,
                            zMid + std::sin(ang) * gatherSpread * 0.35f};

    *halfR = marbleR * 0.5f;
    *drawMerged = static_cast<int>(ph) < static_cast<int>(McStoryPhase::SplitHalves);
    (void)nMarbles;

    Vector3 merged = cluster;
    if (static_cast<int>(ph) <= static_cast<int>(McStoryPhase::UnbundleWrap)) {
        merged = cluster;
    } else if (ph == McStoryPhase::GatherCenter) {
        merged = Vector3Lerp(cluster, gather, EaseInOutCubic(u));
    } else if (ph == McStoryPhase::SplitHalves) {
        merged = gather;
    } else if (ph == McStoryPhase::AssignColumns) {
        merged = gather;
    } else {
        merged = gather;
    }

    float splitAmt = 0.f;
    if (static_cast<int>(ph) > static_cast<int>(McStoryPhase::GatherCenter)) {
        if (ph == McStoryPhase::SplitHalves) {
            splitAmt = EaseInOutCubic(u);
        } else {
            splitAmt = 1.f;
        }
    }

    const float sep = gx * 0.28f * splitAmt;
    Vector3 mSide = {merged.x - sep, merged.y, merged.z};
    Vector3 cSide = {merged.x + sep, merged.y, merged.z};

    Vector3 mEnd = mGrid[row][row];
    mEnd.y += marbleR * 0.95f;
    Vector3 cEnd = cGrid[row][row];
    cEnd.y += marbleR * 0.95f;
    if (ph == McStoryPhase::AssignColumns) {
        const float e = EaseInOutCubic(u);
        mSide = Vector3Lerp(mSide, mEnd, e);
        cSide = Vector3Lerp(cSide, cEnd, e);
    } else if (static_cast<int>(ph) > static_cast<int>(McStoryPhase::AssignColumns)) {
        mSide = mEnd;
        cSide = cEnd;
    }

    *outCenter = merged;
    *outM = mSide;
    *outC = cSide;
}

static Vector3 ComputeBundlerMarblePos(int covIdx, int nMarbles, int nGrid, float ox, float oz, float gx, float gz,
                                       float landY, float amp, float marbleR, BundlerStoryPhase ph, float u,
                                       const Vector3& tray) {
    float bx = 0.f;
    float bz = 0.f;
    BundlerBaseXZ(covIdx, nGrid, nMarbles, ox, oz, gx, gz, &bx, &bz);
    const Vector3 plane = {bx, landY, bz};
    const float rise = amp * 0.42f;
    const Vector3 floodHeld = {tray.x + gx * 0.12f, tray.y + marbleR * 2.08f, tray.z - gz * 0.06f};

    auto stagingLerp = [&](float blend) -> Vector3 {
        if (!DemoIsStagingMarble(covIdx)) {
            return plane;
        }
        const Vector3 sw = BundlerStagingWorld(covIdx, tray, gx, amp);
        const float e = EaseInOutCubic(blend);
        return {bx + (sw.x - bx) * e, landY + (sw.y - landY) * e, bz + (sw.z - bz) * e};
    };

    auto posBeforeWrap = [&]() -> Vector3 {
        if (DemoIsStagingMarble(covIdx)) {
            return stagingLerp(1.f);
        }
        if (covIdx == kDemoSuccessCov) {
            return floodHeld;
        }
        return plane;
    };

    switch (ph) {
        case BundlerStoryPhase::FadeScene:
        case BundlerStoryPhase::Checkmarks:
            return plane;
        case BundlerStoryPhase::RiseStaging:
            return stagingLerp(u);
        case BundlerStoryPhase::SuccessFlood:
            if (DemoIsStagingMarble(covIdx)) {
                return stagingLerp(1.f);
            }
            if (covIdx != kDemoSuccessCov) {
                return plane;
            }
            if (u < 0.24f) {
                const float e = EaseOutCubic(u / 0.24f);
                return {bx, landY + rise * e, bz};
            }
            if (u < 0.62f) {
                return {bx, landY + rise, bz};
            }
            {
                const float e = EaseInOutCubic((u - 0.62f) / (1.f - 0.62f));
                const Vector3 hi = {bx, landY + rise, bz};
                return Vector3Lerp(hi, floodHeld, e);
            }
        case BundlerStoryPhase::FailFloaterDep:
            if (DemoIsStagingMarble(covIdx)) {
                return stagingLerp(1.f);
            }
            if (covIdx == kDemoSuccessCov) {
                return floodHeld;
            }
            if (covIdx != kDemoFailDepCov) {
                return plane;
            }
            if (u < 0.22f) {
                const float e = EaseOutCubic(u / 0.22f);
                return {bx, landY + rise * e, bz};
            }
            if (u < 0.72f) {
                return {bx, landY + rise, bz};
            }
            {
                const float e = EaseInOutCubic((u - 0.72f) / 0.28f);
                return {bx, landY + rise * (1.f - e), bz};
            }
        case BundlerStoryPhase::FailExcessIncompat:
            if (DemoIsStagingMarble(covIdx)) {
                return stagingLerp(1.f);
            }
            if (covIdx == kDemoSuccessCov) {
                return floodHeld;
            }
            if (covIdx != kDemoFailIncompatCov) {
                return plane;
            }
            if (u < 0.22f) {
                const float e = EaseOutCubic(u / 0.22f);
                return {bx, landY + rise * e, bz};
            }
            if (u < 0.72f) {
                return {bx, landY + rise, bz};
            }
            {
                const float e = EaseInOutCubic((u - 0.72f) / 0.28f);
                return {bx, landY + rise * (1.f - e), bz};
            }
        case BundlerStoryPhase::WrapBundle: {
            const Vector3 prev = posBeforeWrap();
            if (!DemoPackageContains(covIdx)) {
                return prev;
            }
            const Vector3 tgt = BundlerPackageCluster(covIdx, tray, gx, gz, marbleR);
            const float e = EaseInOutCubic(u);
            return Vector3Lerp(prev, tgt, e);
        }
        case BundlerStoryPhase::Done:
        default:
            if (DemoPackageContains(covIdx)) {
                return BundlerPackageCluster(covIdx, tray, gx, gz, marbleR);
            }
            return posBeforeWrap();
    }
}

static void DrawStoryFloorGrid(float ox, float oz, float gx, float gz, int n, const Color& col,
                               float lineAlpha = 1.f) {
    if (lineAlpha < 0.004f) {
        return;
    }
    Color c = col;
    c.a = static_cast<unsigned char>(
        std::round(static_cast<float>(col.a) * std::max(0.f, std::min(1.f, lineAlpha))));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const float x0 = ox + i * gx;
            const float z0 = oz + j * gz;
            if (i + 1 < n) {
                DrawLine3D({x0, 0.f, z0}, {ox + (i + 1) * gx, 0.f, z0}, c);
            }
            if (j + 1 < n) {
                DrawLine3D({x0, 0.f, z0}, {x0, 0.f, oz + (j + 1) * gz}, c);
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

    double lambdaLogMin = 0.0;
    double lambdaLogMax = 1.0;
    if (surf.parametricLambda) {
        const double lr = std::max(surf.lambdaLive, 1e-15);
        lambdaLogMin = std::log10(lr) - 3.0;
        lambdaLogMax = std::log10(lr) + 3.0;
        if (!(lambdaLogMax > lambdaLogMin)) {
            lambdaLogMax = lambdaLogMin + 1.0;
        }
    }
    bool lambdaSliderDragging = false;
    const double marginSLogMin = std::log10(0.12);
    const double marginSLogMax = std::log10(18.0);
    bool marginScaleEnabled = false;
    double marginScaleS = 1.0;
    bool marginSliderDragging = false;

    const int sw = 1400;
    const int sh = 900;
    char winTitle[96];
    std::snprintf(winTitle, sizeof(winTitle), "Package %d — QUBO block (single column, n=%d)",
                  surf.packageIndex, surf.n);
    InitWindow(sw, sh, winTitle);
    SetTargetFPS(120);
    InitStoryEquationFont();

    Camera3D camera = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        45.0f,
        CAMERA_PERSPECTIVE,
    };

    double scaleDen = std::max(MaxAbsQ(surf), 1e-12);
    const float amp = static_cast<float>(std::max(surf.n, 3)) * 0.35f;
    const float gx = 1.0f;
    const float gz = 1.0f;
    const float ox = -(surf.n - 1) * gx * 0.5f;
    const float oz = -(surf.n - 1) * gz * 0.5f;

    int qBlockBaseI = 0;
    int qBlockBaseJ = 0;
    int qMeshN = 1;
    ComputeDemoQBlock(surf, &qBlockBaseI, &qBlockBaseJ, &qMeshN);
    double scaleDenBlock =
        std::max(MaxAbsQSubmatrix(surf, qBlockBaseI, qBlockBaseJ, qMeshN), 1e-12);
    const float ampBlock = static_cast<float>(std::max(qMeshN, 3)) * 0.32f;
    /** Match the story floor grid (DrawStoryFloorGrid / marble land), not the denser teal Q tile pitch. */
    const int floorSpanCells = std::max(1, surf.n - 1);
    const float matSpanWorldX = static_cast<float>(floorSpanCells) * gx;
    const float matSpanWorldZ = static_cast<float>(floorSpanCells) * gz;
    /** Stretch qMeshN×qMeshN over the full n×n floor footprint so it replaces the underlying mesh visually. */
    const float gxQBlock =
        (qMeshN > 1) ? (matSpanWorldX / static_cast<float>(qMeshN - 1)) : gx;
    const float gzQBlock =
        (qMeshN > 1) ? (matSpanWorldZ / static_cast<float>(qMeshN - 1)) : gz;
    const float oxBlock = -(qMeshN - 1) * gxQBlock * 0.5f;
    const float ozBlock = -(qMeshN - 1) * gzQBlock * 0.5f;

    // Default orbit: far enough to frame the full n×n story floor (step 0+); not capped to the small Q mesh.
    const float camRadial = static_cast<float>(std::max(surf.n + 6, qMeshN + 8)) * 0.82f;
    camera.position = {camRadial * 1.0f, camRadial * 0.58f, camRadial * 1.0f};

    auto gridPosBlock = [&](int i, int j) -> Vector3 {
        const int gi = qBlockBaseI + i;
        const int gj = qBlockBaseJ + j;
        double qij = surf.qAt(gi, gj);
        float hn = static_cast<float>(qij / scaleDenBlock);
        if (hn > 0.f) {
            hn *= kQBlockPositivePeakGain;
        }
        float y = hn * ampBlock;
        return {oxBlock + static_cast<float>(i) * gxQBlock, y, ozBlock + static_cast<float>(j) * gzQBlock};
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
    float incompatElapsed = 0.f;
    bool incompatPlayEntrance = true;
    float bundlerElapsed = 0.f;
    bool bundlerPlayEntrance = true;
    float mcElapsed = 0.f;
    bool mcPlayEntrance = true;
    float mcMergeElapsed = 0.f;
    bool mcMergePlayEntrance = true;
    float qSliceElapsed = 0.f;
    bool qSlicePlayEntrance = true;
    /** After slice story reaches Done: mesh stays flat until →; then animates to full Q and advances to QBlockField. */
    float qBlockExtrudeU = 0.f;
    bool qBlockExtrudeAnimating = false;
    float qBlockExtrudeElapsed = 0.f;
    float hamiltonianBlendU = 0.f;
    float hamiltonianBlendElapsed = 0.f;
    Vector3 qLabelWorld = {0.f, 0.f, 0.f};
    bool haveQLabelWorld = false;
    std::vector<HamRollingBall> hamRollingBalls;
    float qaoaStoryElapsed = 0.f;
    bool qaoaPlayEntrance = true;
    float qaoaBoxSpinAngle = 0.f;
    int qaoaDiagI = 0;
    int qaoaOffI = 0;
    int qaoaOffJ = 1;
    double qaoaScaleDenSnap = 1.0;
    bool qaoaEnergyAltSnap = false;

    while (!WindowShouldClose()) {
        const QMatrixLayout qLay = MakeQMatrixLayout(surf, sh);
        const Rectangle qHit = QMatrixHitBounds(qLay);
        const Vector2 mouse = GetMousePosition();
        const bool mouseOnQPanel = showQMatrix && CheckCollisionPointRec(mouse, qHit);

        if (surf.parametricLambda && IsKeyPressed(KEY_M)) {
            marginScaleEnabled = !marginScaleEnabled;
            ApplyLambdaToQ(surf, marginScaleEnabled ? marginScaleS : 1.0);
        }
        const double marginScaleEffective = marginScaleEnabled ? marginScaleS : 1.0;

        constexpr float kLambdaSliderScreenX = 20.f;
        const float marginSliderScreenY = static_cast<float>(sh) - 112.f;
        const float lambdaSliderScreenY = static_cast<float>(sh) - 76.f;
        const bool mouseOnLambdaSlider =
            surf.parametricLambda &&
            CheckCollisionPointRec(mouse, ParamSliderHitRect(kLambdaSliderScreenX, lambdaSliderScreenY));
        const bool mouseOnMarginSlider =
            surf.parametricLambda && marginScaleEnabled &&
            CheckCollisionPointRec(mouse, ParamSliderHitRect(kLambdaSliderScreenX, marginSliderScreenY));
        const bool paramUiBlocksOrbit = mouseOnLambdaSlider || lambdaSliderDragging || mouseOnMarginSlider ||
                                        marginSliderDragging;

        if (!mouseOnQPanel && !paramUiBlocksOrbit) {
            StepTrackball(camera);
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && !mouseOnQPanel && !paramUiBlocksOrbit) {
            autoSpin = false;
        }
        if (IsKeyPressed(KEY_R)) autoSpin = !autoSpin;
        if (IsKeyPressed(KEY_I)) showLegend = !showLegend;
        if (IsKeyPressed(KEY_Q)) showQMatrix = !showQMatrix;
        if (IsKeyPressed(KEY_B) && storyStep == StoryStep::HamiltonianLandscape) {
            InitHamRollingBalls(
                hamRollingBalls, qMeshN,
                static_cast<uint32_t>(std::lround(GetTime() * 1.0e6)) ^ 0x9E3779B9u);
        }

        if (surf.parametricLambda) {
            if (!IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
                lambdaSliderDragging = false;
                marginSliderDragging = false;
            } else if (!mouseOnQPanel) {
                const Rectangle lamHit = ParamSliderHitRect(kLambdaSliderScreenX, lambdaSliderScreenY);
                const Rectangle marHit = ParamSliderHitRect(kLambdaSliderScreenX, marginSliderScreenY);
                const Vector2 mpos = GetMousePosition();
                if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                    if (CheckCollisionPointRec(mpos, lamHit)) {
                        lambdaSliderDragging = true;
                    } else if (marginScaleEnabled && CheckCollisionPointRec(mpos, marHit)) {
                        marginSliderDragging = true;
                    }
                }
                if (lambdaSliderDragging) {
                    float t = (mpos.x - kLambdaSliderScreenX) / kLambdaSliderW;
                    surf.lambdaLive = LambdaFromSliderT(t, lambdaLogMin, lambdaLogMax);
                    ApplyLambdaToQ(surf, marginScaleEffective);
                } else if (marginSliderDragging && marginScaleEnabled) {
                    float t = (mpos.x - kLambdaSliderScreenX) / kLambdaSliderW;
                    marginScaleS = LambdaFromSliderT(t, marginSLogMin, marginSLogMax);
                    ApplyLambdaToQ(surf, marginScaleS);
                }
            }
        }
        scaleDen = std::max(MaxAbsQ(surf), 1e-12);
        scaleDenBlock = std::max(MaxAbsQSubmatrix(surf, qBlockBaseI, qBlockBaseJ, qMeshN), 1e-12);
        /** If every x in the block is 1 (or x absent), Q_ij·x_i·x_j == Q_ij — use alternating demo bits so morph is visible. */
        const bool hamMorphUseAlternatingDemo =
            !SurfHasAssignment(surf) || QBlockVarsAllOne(surf, qBlockBaseI, qBlockBaseJ, qMeshN);
        const double termAbsMaxBlock = MaxAbsQEnergyTermSubmatrixForMorph(
            surf, qBlockBaseI, qBlockBaseJ, qMeshN, hamMorphUseAlternatingDemo);
        const double scaleDenMeshHam =
            std::max(scaleDenBlock, std::max(termAbsMaxBlock, 1e-18));

        const StoryStep prevStory = storyStep;
        if (IsKeyPressed(KEY_RIGHT)) {
            if (storyStep == StoryStep::QSliceToBlock) {
                QSliceStoryPhase qRightPh = QSliceStoryPhase::ShowGrid;
                float qRightU = 0.f;
                QSliceDecodePhase(qSlicePlayEntrance ? qSliceElapsed : 1.0e6f, &qRightPh, &qRightU);
                if (qRightPh == QSliceStoryPhase::Done && !qBlockExtrudeAnimating && qBlockExtrudeU < 0.02f) {
                    qBlockExtrudeAnimating = true;
                    qBlockExtrudeElapsed = 0.f;
                }
            } else {
                const int next = std::min(static_cast<int>(storyStep) + 1, static_cast<int>(StoryStep::Count) - 1);
                storyStep = static_cast<StoryStep>(next);
            }
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
        if (storyStep != prevStory && storyStep == StoryStep::MarblesIncompatibilities) {
            incompatPlayEntrance = (prevStory == StoryStep::MarblesDependencies);
            incompatElapsed = 0.f;
        }
        if (storyStep != prevStory && storyStep == StoryStep::MarblesBundler) {
            bundlerPlayEntrance = (prevStory == StoryStep::MarblesIncompatibilities);
            bundlerElapsed = 0.f;
        }
        if (storyStep != prevStory && storyStep == StoryStep::MarblesMcDecompose) {
            mcPlayEntrance = (prevStory == StoryStep::MarblesBundler);
            mcElapsed = 0.f;
        }
        if (storyStep != prevStory && storyStep == StoryStep::MarblesMcMerge) {
            mcMergePlayEntrance = (prevStory == StoryStep::MarblesMcDecompose);
            mcMergeElapsed = 0.f;
        }
        if (storyStep != prevStory && storyStep == StoryStep::QSliceToBlock) {
            qSlicePlayEntrance = (prevStory == StoryStep::MarblesMcMerge);
            qSliceElapsed = 0.f;
            qBlockExtrudeU = 0.f;
            qBlockExtrudeAnimating = false;
            qBlockExtrudeElapsed = 0.f;
        }
        if (storyStep != prevStory && storyStep == StoryStep::HamiltonianLandscape) {
            hamiltonianBlendElapsed = 0.f;
            hamiltonianBlendU = 0.f;
            InitHamRollingBalls(hamRollingBalls, qMeshN, 7u);
        }
        if (storyStep != prevStory && storyStep == StoryStep::QaoaGateDemo) {
            qaoaPlayEntrance = (prevStory == StoryStep::HamiltonianLandscape);
            qaoaStoryElapsed = 0.f;
            qaoaBoxSpinAngle = 0.f;
            qaoaScaleDenSnap = scaleDenMeshHam;
            qaoaEnergyAltSnap = hamMorphUseAlternatingDemo;
            FindQaoaLowestVertices(surf, qBlockBaseI, qBlockBaseJ, qMeshN, qaoaScaleDenSnap, ampBlock,
                                  qaoaEnergyAltSnap, &qaoaDiagI, &qaoaOffI, &qaoaOffJ);
        }
        if (prevStory == StoryStep::QSliceToBlock && storyStep != StoryStep::QSliceToBlock &&
            storyStep != StoryStep::QBlockField) {
            qBlockExtrudeAnimating = false;
            qBlockExtrudeU = 0.f;
            qBlockExtrudeElapsed = 0.f;
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

        if (storyStep != StoryStep::MarblesIncompatibilities) {
            incompatElapsed = 0.f;
        } else if (incompatPlayEntrance) {
            incompatElapsed += GetFrameTime();
        }

        if (storyStep != StoryStep::MarblesBundler) {
            bundlerElapsed = 0.f;
        } else if (bundlerPlayEntrance) {
            bundlerElapsed += GetFrameTime();
        }

        if (storyStep != StoryStep::MarblesMcDecompose) {
            mcElapsed = 0.f;
        } else if (mcPlayEntrance) {
            mcElapsed += GetFrameTime();
        }

        if (storyStep != StoryStep::MarblesMcMerge) {
            mcMergeElapsed = 0.f;
        } else if (mcMergePlayEntrance) {
            mcMergeElapsed += GetFrameTime();
        }

        if (storyStep != StoryStep::QSliceToBlock) {
            qSliceElapsed = 0.f;
        } else if (qSlicePlayEntrance) {
            qSliceElapsed += GetFrameTime();
        }

        if (storyStep != StoryStep::QaoaGateDemo) {
            qaoaStoryElapsed = 0.f;
        } else if (qaoaPlayEntrance) {
            qaoaStoryElapsed += GetFrameTime();
        }

        if (storyStep == StoryStep::QSliceToBlock && qBlockExtrudeAnimating) {
            qBlockExtrudeElapsed += GetFrameTime();
            const float kQBlockExtrudeDur = 1.15f;
            qBlockExtrudeU = SmoothStep(std::min(1.f, qBlockExtrudeElapsed / kQBlockExtrudeDur));
            if (qBlockExtrudeElapsed >= kQBlockExtrudeDur) {
                qBlockExtrudeAnimating = false;
                qBlockExtrudeU = 1.f;
                storyStep = StoryStep::QBlockField;
            }
        }

        if (storyStep == StoryStep::HamiltonianLandscape) {
            hamiltonianBlendElapsed += GetFrameTime();
            const float kHamiltonianMorphDur = 1.35f;
            hamiltonianBlendU = SmoothStep(std::min(1.f, hamiltonianBlendElapsed / kHamiltonianMorphDur));
            if (qMeshN > 1 && !hamRollingBalls.empty()) {
                UpdateHamRollingBalls(hamRollingBalls, surf, qBlockBaseI, qBlockBaseJ, qMeshN, scaleDenMeshHam, ampBlock,
                                      1.f, hamiltonianBlendU, hamMorphUseAlternatingDemo, GetFrameTime());
            }
        } else {
            hamiltonianBlendElapsed = 0.f;
            hamiltonianBlendU = 0.f;
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

        QaoaStoryPhase qaoaPh = QaoaStoryPhase::PickDiagonal;
        float qaoaPhaseU = 0.f;
        Vector3 qaoaWorldD = {0.f, 0.f, 0.f};
        Vector3 qaoaWorldO = {0.f, 0.f, 0.f};
        Vector3 qaoaBoxPos = {0.f, 0.f, 0.f};
        float qaoaBoxW = 1.f;
        float qaoaBoxH = 1.f;
        float qaoaBoxD = 1.f;
        float qaoaMeshAlpha = 0.f;
        float qaoaGateLabelA = 0.f;
        float qaoaBundleAlpha = 0.f;
        float qaoaSphereAD = 0.f;
        float qaoaSphereAO = 0.f;
        float qaoaBoxAlpha = 0.f;

        if (storyStep == StoryStep::QaoaGateDemo && qMeshN > 0) {
            QaoaDecodePhase(qaoaPlayEntrance ? qaoaStoryElapsed : 1.0e6f, &qaoaPh, &qaoaPhaseU);

            const float marbleR = std::max(0.04f, std::min(gxQBlock, gzQBlock) * 0.17f);
            const float riseH = ampBlock * 0.58f;
            float riseEase = 0.f;
            if (static_cast<int>(qaoaPh) > static_cast<int>(QaoaStoryPhase::RiseUp)) {
                riseEase = 1.f;
            } else if (qaoaPh == QaoaStoryPhase::RiseUp) {
                riseEase = EaseInOutCubic(qaoaPhaseU);
            }
            const float yLift = riseH * riseEase;

            qaoaMeshAlpha = 1.f;
            if (qaoaPh == QaoaStoryPhase::GridFade) {
                qaoaMeshAlpha = 1.f - SmoothStep(qaoaPhaseU);
            } else if (static_cast<int>(qaoaPh) > static_cast<int>(QaoaStoryPhase::GridFade)) {
                qaoaMeshAlpha = 0.f;
            }

            const Vector3 pD0 =
                QaoaBallWorld(surf, qBlockBaseI, qBlockBaseJ, qaoaDiagI, qaoaDiagI, qaoaScaleDenSnap, ampBlock,
                              qaoaEnergyAltSnap, oxBlock, ozBlock, gxQBlock, gzQBlock, marbleR, yLift);
            const Vector3 pO0 =
                QaoaBallWorld(surf, qBlockBaseI, qBlockBaseJ, qaoaOffI, qaoaOffJ, qaoaScaleDenSnap, ampBlock,
                              qaoaEnergyAltSnap, oxBlock, ozBlock, gxQBlock, gzQBlock, marbleR, yLift);

            const float midU = 0.5f * static_cast<float>(std::max(0, qMeshN - 1));
            const Vector3 meshMid = {oxBlock + midU * gxQBlock, ampBlock * 0.22f, ozBlock + midU * gzQBlock};
            qaoaBoxW = std::max(gxQBlock, gzQBlock) * 1.55f;
            qaoaBoxH = ampBlock * 0.42f;
            qaoaBoxD = qaoaBoxW * 0.88f;
            qaoaBoxPos = {meshMid.x + qaoaBoxW * 1.2f, meshMid.y + qaoaBoxH * 0.55f, meshMid.z};

            const Vector3 targetD = {qaoaBoxPos.x - qaoaBoxW * 0.48f, qaoaBoxPos.y + qaoaBoxH * 0.05f,
                                     qaoaBoxPos.z + qaoaBoxD * 0.14f};
            const Vector3 targetO = {qaoaBoxPos.x - qaoaBoxW * 0.1f, qaoaBoxPos.y + qaoaBoxH * 0.05f,
                                     qaoaBoxPos.z - qaoaBoxD * 0.2f};
            float dashT = 0.f;
            if (qaoaPh == QaoaStoryPhase::DashToQaoa) {
                dashT = SmoothStep(qaoaPhaseU);
            } else if (static_cast<int>(qaoaPh) > static_cast<int>(QaoaStoryPhase::DashToQaoa)) {
                dashT = 1.f;
            }
            qaoaWorldD = Vector3Lerp(pD0, targetD, dashT);
            qaoaWorldO = Vector3Lerp(pO0, targetO, dashT);

            qaoaSphereAD = 1.f;
            qaoaSphereAO = 1.f;
            if (qaoaPh == QaoaStoryPhase::PickDiagonal) {
                qaoaSphereAO = 0.f;
            }
            if (qaoaPh == QaoaStoryPhase::MorphGates) {
                const float m = SmoothStep(qaoaPhaseU);
                qaoaSphereAD *= (1.f - 0.9f * m);
                qaoaSphereAO *= (1.f - 0.9f * m);
            } else if (qaoaPh == QaoaStoryPhase::DashToQaoa) {
                qaoaSphereAD *= 0.12f;
                qaoaSphereAO *= 0.12f;
            } else if (static_cast<int>(qaoaPh) >= static_cast<int>(QaoaStoryPhase::BoxSpinSpit)) {
                const float ab = (qaoaPh == QaoaStoryPhase::BoxSpinSpit) ? SmoothStep(qaoaPhaseU) : 1.f;
                const float remn = (1.f - ab) * 0.12f;
                qaoaSphereAD *= remn;
                qaoaSphereAO *= remn;
            }

            qaoaBoxAlpha = 0.f;
            if (qaoaPh == QaoaStoryPhase::MorphGates) {
                qaoaBoxAlpha = SmoothStep(qaoaPhaseU) * 0.92f;
            } else if (qaoaPh == QaoaStoryPhase::DashToQaoa) {
                qaoaBoxAlpha = 0.92f + 0.08f * SmoothStep(qaoaPhaseU);
            } else if (static_cast<int>(qaoaPh) >= static_cast<int>(QaoaStoryPhase::BoxSpinSpit)) {
                qaoaBoxAlpha = 1.f;
            }

            qaoaGateLabelA = 0.f;
            if (qaoaPh == QaoaStoryPhase::MorphGates) {
                qaoaGateLabelA = SmoothStep(qaoaPhaseU);
            } else if (qaoaPh == QaoaStoryPhase::DashToQaoa) {
                qaoaGateLabelA = 1.f;
            } else if (qaoaPh == QaoaStoryPhase::BoxSpinSpit) {
                qaoaGateLabelA = std::max(0.f, 1.f - SmoothStep(qaoaPhaseU * 1.12f));
            }

            qaoaBundleAlpha = 0.f;
            if (qaoaPh == QaoaStoryPhase::BoxSpinSpit) {
                qaoaBundleAlpha = SmoothStep((qaoaPhaseU - 0.18f) / 0.75f);
            } else if (qaoaPh == QaoaStoryPhase::HoldOutro) {
                qaoaBundleAlpha = 1.f;
            }

            const float dtSpin = GetFrameTime();
            if (qaoaPh == QaoaStoryPhase::BoxSpinSpit) {
                qaoaBoxSpinAngle +=
                    400.f * dtSpin * (0.52f + 0.48f * std::sin(qaoaPhaseU * 3.14159265f));
            } else if (qaoaPh == QaoaStoryPhase::HoldOutro) {
                qaoaBoxSpinAngle += 22.f * dtSpin;
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        Vector3 mcLabelWorldM = {0.f, 0.f, 0.f};
        Vector3 mcLabelWorldC = {0.f, 0.f, 0.f};
        bool mcHaveMatrixLabels = false;
        haveQLabelWorld = false;
        BeginMode3D(camera);

        if (storyStep == StoryStep::MarblesDropWhite || storyStep == StoryStep::MarblesFamilyColor ||
            storyStep == StoryStep::MarblesDependencies || storyStep == StoryStep::MarblesIncompatibilities ||
            storyStep == StoryStep::MarblesBundler) {
            DrawStoryFloorGrid(ox, oz, gx, gz, surf.n, g_style.storyFloorLine);

            const int nMarbles = nCoverageMarbles;
            const float startY = amp * 9.0f;
            const float landY = amp * 0.14f;
            const float marbleR = 0.052f * amp;
            const double now = GetTime();
            const float rawT = static_cast<float>(now - marbleAnimT0);
            const bool colorByFamily = (storyStep == StoryStep::MarblesFamilyColor ||
                                        storyStep == StoryStep::MarblesDependencies ||
                                        storyStep == StoryStep::MarblesIncompatibilities ||
                                        storyStep == StoryStep::MarblesBundler);

            BundlerStoryPhase bunPh = BundlerStoryPhase::FadeScene;
            float bunU = 0.f;
            Vector3 bundlerTrayCenter = {0.f, 0.f, 0.f};
            if (storyStep == StoryStep::MarblesBundler) {
                BundlerDecodePhase(bundlerPlayEntrance ? bundlerElapsed : 1.0e6f, &bunPh, &bunU);
                bundlerTrayCenter = BundlerTrayCenter(ox, oz, gx, gz, surf.n, landY, amp);
                const float trayFade =
                    (bunPh == BundlerStoryPhase::FadeScene) ? SmoothStep(bunU) : 1.f;
                Color trayFill = g_style.bundlerTray;
                trayFill.a =
                    static_cast<unsigned char>(std::round(static_cast<float>(trayFill.a) * trayFade));
                Color trayWire = g_style.bundlerTrayWire;
                trayWire.a =
                    static_cast<unsigned char>(std::round(static_cast<float>(trayWire.a) * trayFade));
                DrawCube(bundlerTrayCenter, gx * 2.35f, marbleR * 0.42f, gz * 2.05f, trayFill);
                DrawCubeWires(bundlerTrayCenter, gx * 2.35f, marbleR * 0.42f, gz * 2.05f, trayWire);
                if (bunPh == BundlerStoryPhase::WrapBundle || bunPh == BundlerStoryPhase::Done) {
                    const float wrapT = (bunPh == BundlerStoryPhase::WrapBundle) ? bunU : 1.f;
                    const float wrapScale = 0.5f + 0.5f * wrapT;
                    Color wrapCol = g_style.bundleWrap;
                    wrapCol.a = static_cast<unsigned char>(
                        std::round(static_cast<float>(wrapCol.a) * (0.35f + 0.65f * wrapT)));
                    DrawCube(bundlerTrayCenter,
                             gx * 2.35f * wrapScale * 1.08f,
                             marbleR * 2.8f * wrapScale,
                             gz * 2.05f * wrapScale * 1.08f,
                             wrapCol);
                    DrawCubeWires(bundlerTrayCenter,
                                  gx * 2.35f * wrapScale * 1.08f,
                                  marbleR * 2.8f * wrapScale,
                                  gz * 2.05f * wrapScale * 1.08f,
                                  trayWire);
                }
            }

            float depSpreadT = 0.f;
            if (storyStep == StoryStep::MarblesDependencies) {
                if (!depPlayEntrance) {
                    depSpreadT = 1.f;
                } else {
                    depSpreadT = EaseInOutCubic(std::min(1.f, depSpreadElapsed / kDepSpreadSec));
                }
            } else if (storyStep == StoryStep::MarblesIncompatibilities) {
                depSpreadT = 1.f;
            }

            float incompatT = 0.f;
            if (storyStep == StoryStep::MarblesIncompatibilities) {
                if (!incompatPlayEntrance) {
                    incompatT = 1.f;
                } else {
                    incompatT = EaseInOutCubic(std::min(1.f, incompatElapsed / kIncompatDrawSec));
                }
            }

            float familyColorBlend = 0.f;
            float familyGroupMove = 0.f;
            if (colorByFamily) {
                if (storyStep == StoryStep::MarblesDependencies ||
                    storyStep == StoryStep::MarblesIncompatibilities || storyStep == StoryStep::MarblesBundler) {
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
                Vector3 bundlerPos = {0.f, 0.f, 0.f};
                if (storyStep == StoryStep::MarblesBundler) {
                    bundlerPos = ComputeBundlerMarblePos(covIdx, nMarbles, surf.n, ox, oz, gx, gz, landY, amp, marbleR,
                                                         bunPh, bunU, bundlerTrayCenter);
                    lx = bundlerPos.x;
                    lz = bundlerPos.z;
                } else if (storyStep == StoryStep::MarblesDependencies ||
                           storyStep == StoryStep::MarblesIncompatibilities) {
                    float cx = 0.f;
                    float cz = 0.f;
                    float sx = 0.f;
                    float sz = 0.f;
                    CoverageGroupLandXZ(covIdx, surf.n, nMarbles, ox, oz, gx, gz, &cx, &cz);
                    CoverageSpreadLandXZ(covIdx, surf.n, nMarbles, ox, oz, gx, gz, &sx, &sz);
                    float spreadMul = 1.f;
                    if (storyStep == StoryStep::MarblesIncompatibilities) {
                        spreadMul = 1.f + 0.68f * incompatT;
                    }
                    lx = cx + (sx - cx) * depSpreadT * spreadMul;
                    lz = cz + (sz - cz) * depSpreadT * spreadMul;
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
                float y = startY + (landY - startY) * e;
                if (storyStep == StoryStep::MarblesBundler) {
                    y = bundlerPos.y;
                }
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

                if (storyStep == StoryStep::MarblesBundler && static_cast<int>(bunPh) >= static_cast<int>(BundlerStoryPhase::Checkmarks) &&
                    covIdx < nMarbles && DemoPackageContains(covIdx)) {
                    const float ckFade =
                        (bunPh == BundlerStoryPhase::Checkmarks) ? SmoothStep(bunU) : 1.f;
                    Color ck = g_style.checkAccent;
                    ck.a = static_cast<unsigned char>(std::round(static_cast<float>(ck.a) * ckFade));
                    const Vector3 tip = {marbleCenter.x, marbleCenter.y + marbleR * 2.15f, marbleCenter.z};
                    DrawCheckmark3D(tip, marbleR * 1.35f, ck);
                }
            }

            const bool showDepArrows =
                (storyStep == StoryStep::MarblesDependencies && depSpreadT > 0.02f) ||
                (storyStep == StoryStep::MarblesIncompatibilities) ||
                (storyStep == StoryStep::MarblesBundler);
            if (showDepArrows && nMarbles > 0) {
                Color ac = g_style.depArrow;
                float depAlpha = 1.f;
                if (storyStep == StoryStep::MarblesDependencies) {
                    depAlpha = std::min(1.f, depSpreadT * 1.05f);
                } else if (storyStep == StoryStep::MarblesBundler) {
                    depAlpha = 0.09f;
                }
                ac.a = static_cast<unsigned char>(std::round(static_cast<float>(ac.a) * depAlpha));
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

            if (storyStep == StoryStep::MarblesIncompatibilities && incompatT > 0.02f && nMarbles > 0) {
                Color ic = g_style.incompatLine;
                ic.a = static_cast<unsigned char>(
                    std::round(static_cast<float>(ic.a) * std::min(1.f, incompatT * 1.05f)));
                for (const CoverageIncompatiblePair& ip : kCoverageIncompatiblePairs) {
                    if (ip.a < 0 || ip.b < 0 || ip.a >= nMarbles || ip.b >= nMarbles) {
                        continue;
                    }
                    if (CoverageIsMandatoryFamilyPick(ip.a) || CoverageIsMandatoryFamilyPick(ip.b)) {
                        continue;
                    }
                    DrawIncompatibleBrokenLineBetweenMarbles(marbleCenterForCov[static_cast<size_t>(ip.a)],
                                                             marbleCenterForCov[static_cast<size_t>(ip.b)], marbleR,
                                                             ic);
                }
            }

            if (storyStep == StoryStep::MarblesBundler && nMarbles > 0) {
                Color icBg = g_style.incompatLine;
                icBg.a = static_cast<unsigned char>(std::round(static_cast<float>(icBg.a) * 0.1f));
                for (const CoverageIncompatiblePair& ip : kCoverageIncompatiblePairs) {
                    if (ip.a < 0 || ip.b < 0 || ip.a >= nMarbles || ip.b >= nMarbles) {
                        continue;
                    }
                    if (CoverageIsMandatoryFamilyPick(ip.a) || CoverageIsMandatoryFamilyPick(ip.b)) {
                        continue;
                    }
                    DrawIncompatibleBrokenLineBetweenMarbles(marbleCenterForCov[static_cast<size_t>(ip.a)],
                                                             marbleCenterForCov[static_cast<size_t>(ip.b)], marbleR,
                                                             icBg);
                }

                if (bunPh == BundlerStoryPhase::SuccessFlood && kDemoSuccessReq < nMarbles &&
                    kDemoSuccessCov < nMarbles && bunU >= 0.24f && bunU < 0.62f) {
                    Color ac = g_style.depArrow;
                    const float w = (bunU - 0.24f) / (0.62f - 0.24f);
                    const float flash = BundlerFlashTwice(std::max(0.f, std::min(1.f, w)));
                    ac.a = static_cast<unsigned char>(std::round(static_cast<float>(ac.a) * flash));
                    DrawDependencyArrowBetweenMarbles(
                        marbleCenterForCov[static_cast<size_t>(kDemoSuccessReq)],
                        marbleCenterForCov[static_cast<size_t>(kDemoSuccessCov)], marbleR, ac);
                }

                if (bunPh == BundlerStoryPhase::FailFloaterDep && kDemoFailDepCov < nMarbles &&
                    kDemoFailDepReq < nMarbles) {
                    if (bunU >= 0.22f && bunU < 0.68f) {
                        Color ac = g_style.depArrow;
                        const float w = (bunU - 0.22f) / (0.68f - 0.22f);
                        const float flash = BundlerFlashTwice(std::max(0.f, std::min(1.f, w)));
                        ac.a = static_cast<unsigned char>(std::round(static_cast<float>(ac.a) * flash));
                        DrawDependencyArrowBetweenMarbles(
                            marbleCenterForCov[static_cast<size_t>(kDemoFailDepReq)],
                            marbleCenterForCov[static_cast<size_t>(kDemoFailDepCov)], marbleR, ac);
                    }
                    if (bunU > 0.34f && bunU < 0.62f) {
                        Vector3 mid = Vector3Lerp(marbleCenterForCov[static_cast<size_t>(kDemoFailDepReq)],
                                                  marbleCenterForCov[static_cast<size_t>(kDemoFailDepCov)], 0.52f);
                        mid.y += marbleR * 1.85f;
                        const Color rx = {235, 55, 55, 240};
                        DrawMiniRedX3D(mid, marbleR * 0.55f, rx);
                    }
                }

                if (bunPh == BundlerStoryPhase::FailExcessIncompat && kDemoFailIncompatCov < nMarbles &&
                    kDemoIncompatPartner < nMarbles && bunU >= 0.22f && bunU < 0.72f) {
                    Color icf = g_style.incompatLine;
                    const float w = (bunU - 0.22f) / (0.72f - 0.22f);
                    const float flash = BundlerFlashTwice(std::max(0.f, std::min(1.f, w)));
                    icf.a = static_cast<unsigned char>(std::round(static_cast<float>(icf.a) * flash));
                    DrawIncompatibleBrokenLineBetweenMarbles(
                        marbleCenterForCov[static_cast<size_t>(kDemoFailIncompatCov)],
                        marbleCenterForCov[static_cast<size_t>(kDemoIncompatPartner)], marbleR, icf);
                }
            }
        } else if (storyStep == StoryStep::MarblesMcDecompose) {
            DrawStoryFloorGrid(ox, oz, gx, gz, surf.n, g_style.storyFloorLine);

            const int nMarbles = nCoverageMarbles;
            const float landY = amp * 0.14f;
            const float marbleR = 0.052f * amp;

            McStoryPhase mcPh = McStoryPhase::FadeNonPackage;
            float mcU = 0.f;
            McDecodePhase(mcPlayEntrance ? mcElapsed : 1.0e6f, &mcPh, &mcU);

            const Vector3 tray = BundlerTrayCenter(ox, oz, gx, gz, surf.n, landY, amp);
            float mcColStep = gx * 0.44f;
            float mcRowStep = gx * 1.02f;
            Vector3 mGrid[kMcMatrixDim][kMcMatrixDim];
            Vector3 cGrid[kMcMatrixDim][kMcMatrixDim];
            McMatrixGrids(landY, gx, gz, oz, surf.n, &mcColStep, &mcRowStep, mGrid, cGrid);

            float matAlpha = 0.f;
            if (static_cast<int>(mcPh) >= static_cast<int>(McStoryPhase::ShowMatrices)) {
                if (mcPh == McStoryPhase::ShowMatrices) {
                    matAlpha = SmoothStep(mcU);
                } else {
                    matAlpha = 1.f;
                }
            }
            if (matAlpha > 0.02f) {
                const Color coldCGreen = {0, 48, 34, 255};
                const Color hotCGreen = g_style.depArrow;
                DrawMcMatrixPads(mGrid, cGrid, mcColStep, mcRowStep, kMcRowM, kMcRowC, coldCGreen, hotCGreen, matAlpha);
                Color frameM = {120, 175, 220, 255};
                Color frameC = {40, static_cast<unsigned char>(std::min(255, g_style.depArrow.g + 40)),
                                static_cast<unsigned char>(std::min(255, g_style.depArrow.b + 30)), 255};
                DrawMcMatrixFrames(mGrid, mcColStep, mcRowStep, frameM, matAlpha);
                DrawMcMatrixFrames(cGrid, mcColStep, mcRowStep, frameC, matAlpha);
                const Color ballColdM = {20, 55, 110, 255};
                const Color ballHotM = {0, 102, 204, 255};
                const Color ballColdC = {0, 55, 40, 255};
                const Color ballWireM = {150, 210, 255, 220};
                Color ballWireC = g_style.depArrow;
                ballWireC.a = 220;
                DrawMcMatrixCellBalls(mGrid, mcColStep, mcRowStep, kMcRowM, ballColdM, ballHotM, ballWireM, gx,
                                      matAlpha);
                DrawMcMatrixCellBalls(cGrid, mcColStep, mcRowStep, kMcRowC, ballColdC, hotCGreen, ballWireC, gx,
                                      matAlpha);
                mcHaveMatrixLabels = true;
                Vector3 bm = McMatrixGridCenter(mGrid);
                Vector3 bc = McMatrixGridCenter(cGrid);
                const float lift = amp * 0.52f;
                mcLabelWorldM = {bm.x, bm.y + lift, bm.z};
                mcLabelWorldC = {bc.x, bc.y + lift, bc.z};
            }

            float wrapAlpha = 0.f;
            if (static_cast<int>(mcPh) <= static_cast<int>(McStoryPhase::ShowMatrices)) {
                wrapAlpha = 1.f;
            } else if (mcPh == McStoryPhase::UnbundleWrap) {
                wrapAlpha = 1.f - SmoothStep(mcU);
            }
            if (wrapAlpha > 0.02f) {
                Color trayFill = g_style.bundlerTray;
                trayFill.a = static_cast<unsigned char>(
                    std::round(static_cast<float>(trayFill.a) * wrapAlpha));
                DrawCube(tray, gx * 2.35f, marbleR * 0.42f, gz * 2.05f, trayFill);
                DrawCubeWires(tray, gx * 2.35f, marbleR * 0.42f, gz * 2.05f, g_style.bundlerTrayWire);
                Color wrapCol = g_style.bundleWrap;
                wrapCol.a = static_cast<unsigned char>(std::round(
                    static_cast<float>(wrapCol.a) * (0.35f + 0.65f) * wrapAlpha));
                DrawCube(tray, gx * 2.35f * 1.08f, marbleR * 2.8f, gz * 2.05f * 1.08f, wrapCol);
                DrawCubeWires(tray, gx * 2.35f * 1.08f, marbleR * 2.8f, gz * 2.05f * 1.08f,
                              g_style.bundlerTrayWire);
            }

            float fadeOther = 0.f;
            if (mcPh == McStoryPhase::FadeNonPackage) {
                fadeOther = 1.f - SmoothStep(mcU);
            }

            const Color mFill = {0, 102, 204, 210};
            const Color mWire = {140, 200, 255, 255};
            Color cFill = g_style.depArrow;
            cFill.a = 210;
            Color cWire = g_style.depArrow;
            cWire.r = static_cast<unsigned char>(std::min(255, static_cast<int>(cWire.r) + 100));
            cWire.g = static_cast<unsigned char>(std::min(255, static_cast<int>(cWire.g) + 55));
            cWire.b = static_cast<unsigned char>(std::min(255, static_cast<int>(cWire.b) + 45));

            for (int dropSlot = 0; dropSlot < nMarbles; dropSlot++) {
                const size_t ord = static_cast<size_t>(dropSlot);
                const int covIdx =
                    (ord < marbleDropOrder.size()) ? marbleDropOrder[ord] : dropSlot;

                if (!DemoPackageContains(covIdx)) {
                    if (fadeOther < 0.02f) {
                        continue;
                    }
                    const Vector3 p =
                        McNonPackagePlanePos(covIdx, nMarbles, surf.n, ox, oz, gx, gz, landY);
                    const Color fam = CoverageFamilyColorBundlingHtml(covIdx);
                    Color fill = fam;
                    fill.a = static_cast<unsigned char>(std::round(static_cast<float>(fill.a) * fadeOther));
                    Color w = fam;
                    w.a = static_cast<unsigned char>(std::round(static_cast<float>(w.a) * fadeOther));
                    DrawSphere(p, marbleR * 0.88f, fill);
                    DrawSphereWires(p, marbleR, 8, 12, w);
                    continue;
                }

                Vector3 merged = {0.f, 0.f, 0.f};
                Vector3 posM = {0.f, 0.f, 0.f};
                Vector3 posC = {0.f, 0.f, 0.f};
                float halfR = marbleR * 0.5f;
                bool drawMerged = true;
                ComputeMcPackageMarble(covIdx, nMarbles, surf.n, ox, oz, gx, gz, landY, amp, marbleR, mcPh, mcU,
                                       tray, mGrid, cGrid, &merged, &posM, &posC, &halfR, &drawMerged);

                const Color fam = CoverageFamilyColorBundlingHtml(covIdx);
                if (drawMerged) {
                    DrawSphere(merged, marbleR * 0.88f, fam);
                    DrawSphereWires(merged, marbleR, 8, 12, fam);
                } else {
                    Color mf = mFill;
                    Color cf = cFill;
                    mf.r = static_cast<unsigned char>(std::min(255, (fam.r + mf.r) / 2));
                    mf.g = static_cast<unsigned char>(std::min(255, (fam.g + mf.g) / 2));
                    mf.b = static_cast<unsigned char>(std::min(255, (fam.b + mf.b) / 2));
                    cf.r = static_cast<unsigned char>(std::min(255, (fam.r + cf.r) / 2));
                    cf.g = static_cast<unsigned char>(std::min(255, (fam.g + cf.g) / 2));
                    cf.b = static_cast<unsigned char>(std::min(255, (fam.b + cf.b) / 2));
                    DrawSphere(posM, halfR * 0.9f, mf);
                    DrawSphereWires(posM, halfR, 8, 12, mWire);
                    DrawSphere(posC, halfR * 0.9f, cf);
                    DrawSphereWires(posC, halfR, 8, 12, cWire);
                }
            }
        } else if (storyStep == StoryStep::MarblesMcMerge) {
            DrawStoryFloorGrid(ox, oz, gx, gz, surf.n, g_style.storyFloorLine);

            const int nMarbles = nCoverageMarbles;
            const float landY = amp * 0.14f;
            const float marbleR = 0.052f * amp;

            McMergeStoryPhase mph = McMergeStoryPhase::FadeOverlays;
            float mu = 0.f;
            McMergeDecodePhase(mcMergePlayEntrance ? mcMergeElapsed : 1.0e6f, &mph, &mu);

            McStoryPhase mcPh = McStoryPhase::Done;
            float mcUDone = 1.f;
            McDecodePhase(1.0e6f, &mcPh, &mcUDone);

            const Vector3 tray = BundlerTrayCenter(ox, oz, gx, gz, surf.n, landY, amp);
            float mcColStep = gx * 0.44f;
            float mcRowStep = gx * 1.02f;
            Vector3 mGrid[kMcMatrixDim][kMcMatrixDim];
            Vector3 cGrid[kMcMatrixDim][kMcMatrixDim];
            McMatrixGrids(landY, gx, gz, oz, surf.n, &mcColStep, &mcRowStep, mGrid, cGrid);

            const Color coldCGreen = {0, 48, 34, 255};
            const Color hotCGreen = g_style.depArrow;

            float marbleFade = 0.f;
            if (mph == McMergeStoryPhase::FadeOverlays) {
                marbleFade = 1.f - SmoothStep(mu);
            }

            float lift7 = 0.f;
            if (mph == McMergeStoryPhase::LiftCSeven) {
                lift7 = amp * 0.48f * EaseInOutCubic(mu);
            } else if (static_cast<int>(mph) > static_cast<int>(McMergeStoryPhase::FadeOverlays)) {
                lift7 = amp * 0.48f;
            }

            float cross7 = 1.f;
            float crossF = 0.f;
            if (mph == McMergeStoryPhase::ExpandFullStack) {
                crossF = SmoothStep(mu);
                cross7 = 1.f - crossF;
            } else if (static_cast<int>(mph) > static_cast<int>(McMergeStoryPhase::ExpandFullStack)) {
                cross7 = 0.f;
                crossF = 1.f;
            }

            float mergeU = 0.f;
            const float stackH = amp * 0.42f;
            if (mph == McMergeStoryPhase::MergeTeal) {
                mergeU = EaseInOutCubic(mu);
            } else if (mph == McMergeStoryPhase::Done) {
                mergeU = 1.f;
            }

            const float yCfull = landY + stackH * (1.f - mergeU);
            const float zMid = oz + gz * static_cast<float>(std::max(0, surf.n - 1)) * 0.5f;
            const float cxFull = 0.f;
            const float cellQMerge =
                std::max(gx * 0.1f, 9.8f * gx / static_cast<float>(std::max(surf.n, 1)));
            const float thickFull = gx * 0.026f;

            Vector3 cLift[kMcMatrixDim][kMcMatrixDim];
            McCopyGridLiftC(cGrid, lift7, cLift);

            if (cross7 > 0.02f) {
                const float a7 = cross7;
                DrawMcMatrixPads(mGrid, cLift, mcColStep, mcRowStep, kMcRowM, kMcRowC, coldCGreen, hotCGreen, a7);
                Color frameM = {120, 175, 220, 255};
                Color frameC = {40, static_cast<unsigned char>(std::min(255, g_style.depArrow.g + 40)),
                                static_cast<unsigned char>(std::min(255, g_style.depArrow.b + 30)), 255};
                DrawMcMatrixFrames(mGrid, mcColStep, mcRowStep, frameM, a7);
                DrawMcMatrixFrames(cLift, mcColStep, mcRowStep, frameC, a7);
                const float ballA = marbleFade * cross7;
                if (ballA > 0.02f) {
                    const Color ballColdM = {20, 55, 110, 255};
                    const Color ballHotM = {0, 102, 204, 255};
                    const Color ballColdC = {0, 55, 40, 255};
                    const Color ballWireM = {150, 210, 255, 220};
                    Color ballWireC = g_style.depArrow;
                    ballWireC.a = 220;
                    DrawMcMatrixCellBalls(mGrid, mcColStep, mcRowStep, kMcRowM, ballColdM, ballHotM, ballWireM, gx,
                                          ballA);
                    DrawMcMatrixCellBalls(cLift, mcColStep, mcRowStep, kMcRowC, ballColdC, hotCGreen, ballWireC, gx,
                                          ballA);
                }
            }

            if (crossF > 0.02f) {
                const Color mColdB = {15, 40, 85, 255};
                const Color mHotB = {0, 102, 204, 255};
                const Color tCold = {0, 88, 105, 255};
                const Color tHot = {25, 188, 185, 255};
                Color drawMCold = LerpColorRgb(mColdB, tCold, mergeU);
                Color drawMHot = LerpColorRgb(mHotB, tHot, mergeU);
                Color drawCCold = LerpColorRgb(coldCGreen, tCold, mergeU);
                Color drawCHot = LerpColorRgb(hotCGreen, tHot, mergeU);
                Color wireBaseM = {120, 175, 220, 255};
                Color wireBaseC = {50, static_cast<unsigned char>(std::min(255, g_style.depArrow.g + 35)),
                                   static_cast<unsigned char>(std::min(255, g_style.depArrow.b + 25)), 255};
                Color wireTeal = {95, 210, 205, 255};
                Color wM = LerpColorRgb(wireBaseM, wireTeal, mergeU);
                Color wC = LerpColorRgb(wireBaseC, wireTeal, mergeU);
                const float aF = crossF;
                if (mergeU >= 0.997f) {
                    DrawInsuranceCoefficientMatrix(surf.n, surf.n, cxFull, zMid, landY, cellQMerge, cellQMerge, thickFull,
                                                   tCold, tHot, wireTeal, aF, qMeshN);
                    haveQLabelWorld = true;
                    qLabelWorld = {cxFull, landY + amp * 0.5f, zMid};
                } else {
                    DrawInsuranceCoefficientMatrix(surf.n, surf.n, cxFull, zMid, landY, cellQMerge, cellQMerge, thickFull,
                                                   drawMCold, drawMHot, wM, aF, 0);
                    DrawInsuranceCoefficientMatrix(surf.n, surf.n, cxFull, zMid, yCfull, cellQMerge, cellQMerge, thickFull,
                                                   drawCCold, drawCHot, wC, aF, 0);
                }
            }

            if (marbleFade > 0.02f) {
                const Color mFill = {0, 102, 204, 210};
                const Color mWire = {140, 200, 255, 255};
                Color cFill = g_style.depArrow;
                cFill.a = 210;
                Color cWire = g_style.depArrow;
                cWire.r = static_cast<unsigned char>(std::min(255, static_cast<int>(cWire.r) + 100));
                cWire.g = static_cast<unsigned char>(std::min(255, static_cast<int>(cWire.g) + 55));
                cWire.b = static_cast<unsigned char>(std::min(255, static_cast<int>(cWire.b) + 45));

                for (int dropSlot = 0; dropSlot < nMarbles; dropSlot++) {
                    const size_t ord = static_cast<size_t>(dropSlot);
                    const int covIdx =
                        (ord < marbleDropOrder.size()) ? marbleDropOrder[ord] : dropSlot;
                    if (!DemoPackageContains(covIdx)) {
                        continue;
                    }
                    Vector3 merged = {0.f, 0.f, 0.f};
                    Vector3 posM = {0.f, 0.f, 0.f};
                    Vector3 posC = {0.f, 0.f, 0.f};
                    float halfR = marbleR * 0.5f;
                    bool drawMerged = true;
                    ComputeMcPackageMarble(covIdx, nMarbles, surf.n, ox, oz, gx, gz, landY, amp, marbleR, mcPh,
                                           mcUDone, tray, mGrid, cGrid, &merged, &posM, &posC, &halfR, &drawMerged);
                    const Color fam = CoverageFamilyColorBundlingHtml(covIdx);
                    auto applyFade = [marbleFade](Color x) {
                        x.a = static_cast<unsigned char>(
                            std::round(static_cast<float>(x.a) * marbleFade));
                        return x;
                    };
                    if (drawMerged) {
                        Color fill = applyFade(fam);
                        Color w = applyFade(fam);
                        DrawSphere(merged, marbleR * 0.88f, fill);
                        DrawSphereWires(merged, marbleR, 8, 12, w);
                    } else {
                        Color mf = mFill;
                        Color cf = cFill;
                        mf.r = static_cast<unsigned char>(std::min(255, (fam.r + mf.r) / 2));
                        mf.g = static_cast<unsigned char>(std::min(255, (fam.g + mf.g) / 2));
                        mf.b = static_cast<unsigned char>(std::min(255, (fam.b + mf.b) / 2));
                        cf.r = static_cast<unsigned char>(std::min(255, (fam.r + cf.r) / 2));
                        cf.g = static_cast<unsigned char>(std::min(255, (fam.g + cf.g) / 2));
                        cf.b = static_cast<unsigned char>(std::min(255, (fam.b + cf.b) / 2));
                        mf = applyFade(mf);
                        cf = applyFade(cf);
                        Color mw = applyFade(mWire);
                        Color cw = applyFade(cWire);
                        DrawSphere(posM, halfR * 0.9f, mf);
                        DrawSphereWires(posM, halfR, 8, 12, mw);
                        DrawSphere(posC, halfR * 0.9f, cf);
                        DrawSphereWires(posC, halfR, 8, 12, cw);
                    }
                }
            }
        } else if (storyStep == StoryStep::QSliceToBlock) {
            QSliceStoryPhase qph = QSliceStoryPhase::ShowGrid;
            float qu = 0.f;
            QSliceDecodePhase(qSlicePlayEntrance ? qSliceElapsed : 1.0e6f, &qph, &qu);

            float storyFloorA = 1.f;
            if (qph == QSliceStoryPhase::Fade2D) {
                storyFloorA = 1.f - SmoothStep(qu);
            } else if (qph == QSliceStoryPhase::Done) {
                storyFloorA = 0.f;
            }
            DrawStoryFloorGrid(ox, oz, gx, gz, surf.n, g_style.storyFloorLine, storyFloorA);

            const float landYQ = amp * 0.14f;
            const float zMidQ = oz + gz * static_cast<float>(std::max(0, surf.n - 1)) * 0.5f;
            const float cxFullQ = 0.f;
            const float cellQN =
                std::max(gx * 0.1f, 9.8f * gx / static_cast<float>(std::max(surf.n, 1)));
            const float thickQ = gx * 0.024f;
            const Color tColdQ = {0, 88, 105, 255};
            const Color tHotQ = {25, 188, 185, 255};
            Color wireTealQ = {95, 210, 205, 255};

            const int ts = std::max(1, std::min(qMeshN, surf.n));
            const int selTI = qBlockBaseI / ts;
            const int selTJ = qBlockBaseJ / ts;

            float alphaO = 1.f;
            float alphaS = 1.f;
            float hi = 0.f;
            float morph = 0.f;
            float meshA = 0.f;
            const float meshExtrude = qBlockExtrudeU;
            float planeA = 1.f;
            float sliceLineA = 1.f;

            if (qph == QSliceStoryPhase::HighlightSel) {
                hi = 0.14f * (0.5f + 0.5f * std::sin(qu * 6.2831853f * 2.2f));
            } else if (qph == QSliceStoryPhase::FadeOthers) {
                alphaO = 1.f - SmoothStep(qu);
                alphaS = 1.f;
            } else if (qph == QSliceStoryPhase::ExpandMergeMesh) {
                alphaO = 0.f;
                alphaS = 1.f;
                morph = SmoothStep(qu);
                meshA = SmoothStep(std::max(0.f, (qu - 0.1f) / 0.9f));
            } else if (qph == QSliceStoryPhase::Fade2D) {
                alphaO = 0.f;
                alphaS = 1.f;
                morph = 1.f;
                meshA = 1.f;
                planeA = 1.f - SmoothStep(qu);
                sliceLineA = 0.f;
            } else if (qph == QSliceStoryPhase::Done) {
                alphaO = 0.f;
                alphaS = 1.f;
                morph = 1.f;
                meshA = 1.f;
                planeA = 0.f;
                sliceLineA = 0.f;
            }

            if (planeA > 0.004f) {
                DrawTealQMatrixDynamic(surf.n, cxFullQ, zMidQ, landYQ, cellQN, cellQN, thickQ, tColdQ, tHotQ,
                                       wireTealQ, planeA, ts, selTI, selTJ, alphaO, alphaS, hi, qBlockBaseI,
                                       qBlockBaseJ, qMeshN, morph, oxBlock, ozBlock, gxQBlock, gzQBlock);
            }
            if (sliceLineA > 0.02f && surf.n > ts) {
                Color cut = {210, 245, 242, 255};
                DrawQBlockSliceLines(surf.n, cxFullQ, zMidQ, landYQ, cellQN, cellQN, ts, cut,
                                     sliceLineA * planeA);
            }
            if (meshA > 0.02f) {
                DrawQBlockMesh3D(surf, qBlockBaseI, qBlockBaseJ, qMeshN, oxBlock, ozBlock, gxQBlock, gzQBlock,
                                 scaleDenBlock, ampBlock, meshA, meshA, meshA, meshExtrude);
            }

            haveQLabelWorld = true;
            qLabelWorld = {cxFullQ, landYQ + amp * 0.5f, zMidQ};
        } else if (storyStep == StoryStep::QBlockField) {
            // Story floor off here so the n×n grid does not clash with the expanded Q-block mesh.
            DrawQBlockMesh3D(surf, qBlockBaseI, qBlockBaseJ, qMeshN, oxBlock, ozBlock, gxQBlock, gzQBlock,
                             scaleDenBlock, ampBlock, 1.f, 1.f, 1.f, 1.f, 0.f);
        } else if (storyStep == StoryStep::HamiltonianLandscape) {
            DrawQBlockMesh3D(surf, qBlockBaseI, qBlockBaseJ, qMeshN, oxBlock, ozBlock, gxQBlock, gzQBlock,
                             scaleDenMeshHam, ampBlock, 1.f, 1.f, 1.f, 1.f, hamiltonianBlendU,
                             hamMorphUseAlternatingDemo);
            if (qMeshN > 1 && !hamRollingBalls.empty()) {
                const float marbleR = std::min(gxQBlock, gzQBlock) * 0.17f;
                for (const auto& b : hamRollingBalls) {
                    const float ySurf = SampleQBlockMeshHeightBilinear(
                        surf, qBlockBaseI, qBlockBaseJ, qMeshN, b.u, b.v, scaleDenMeshHam, ampBlock, 1.f,
                        hamiltonianBlendU, hamMorphUseAlternatingDemo);
                    const Vector3 c = {oxBlock + b.u * gxQBlock, ySurf + marbleR, ozBlock + b.v * gzQBlock};
                    DrawSphere(c, marbleR * 0.88f, b.fill);
                    DrawSphereWires(c, marbleR, 8, 12, b.wire);
                }
            }
        } else if (storyStep == StoryStep::QaoaGateDemo) {
            if (qMeshN > 0 && qaoaMeshAlpha > 0.02f) {
                DrawQBlockMesh3D(surf, qBlockBaseI, qBlockBaseJ, qMeshN, oxBlock, ozBlock, gxQBlock, gzQBlock,
                                 qaoaScaleDenSnap, ampBlock, qaoaMeshAlpha, qaoaMeshAlpha * 0.88f, 0.f, 1.f, 1.f,
                                 qaoaEnergyAltSnap);
            }
            if (qMeshN > 0) {
                const float marbleRDraw = std::max(0.04f, std::min(gxQBlock, gzQBlock) * 0.17f);
                const auto drawMarbleA = [&](Vector3 c, float a, Color fill, Color wire) {
                    if (a < 0.02f) {
                        return;
                    }
                    Color f = fill;
                    Color w = wire;
                    f.a = static_cast<unsigned char>(std::round(static_cast<float>(f.a) * a));
                    w.a = static_cast<unsigned char>(std::round(static_cast<float>(w.a) * a));
                    DrawSphere(c, marbleRDraw * 0.88f, f);
                    DrawSphereWires(c, marbleRDraw, 8, 12, w);
                };
                const Color fillD = {35, 120, 220, 230};
                const Color wireD = {140, 210, 255, 255};
                const Color fillO = {35, 140, 85, 230};
                const Color wireO = {160, 240, 200, 255};
                drawMarbleA(qaoaWorldD, qaoaSphereAD, fillD, wireD);
                drawMarbleA(qaoaWorldO, qaoaSphereAO, fillO, wireO);
                if (qaoaPh == QaoaStoryPhase::PickDiagonal || qaoaPh == QaoaStoryPhase::PickOffDiagonal) {
                    const float pulse = 1.f + 0.14f * std::sin(static_cast<float>(GetTime()) * 6.8f);
                    Color ring = {255, 230, 120, 200};
                    if (qaoaPh == QaoaStoryPhase::PickDiagonal || qaoaSphereAD > 0.02f) {
                        ring.a = static_cast<unsigned char>(
                            std::round(static_cast<float>(ring.a) * qaoaSphereAD));
                        DrawSphereWires(qaoaWorldD, marbleRDraw * pulse, 10, 14, ring);
                    }
                    if (qaoaPh == QaoaStoryPhase::PickOffDiagonal && qaoaSphereAO > 0.02f) {
                        ring.a = static_cast<unsigned char>(
                            std::round(static_cast<float>(ring.a) * qaoaSphereAO));
                        DrawSphereWires(qaoaWorldO, marbleRDraw * pulse, 10, 14, ring);
                    }
                }
                if (qaoaBoxAlpha > 0.02f) {
                    Color boxFill = {18, 44, 92, static_cast<unsigned char>(qaoaBoxAlpha * 240.f)};
                    Color boxWire = {0, 102, 204, static_cast<unsigned char>(qaoaBoxAlpha * 255.f)};
                    rlPushMatrix();
                    rlTranslatef(qaoaBoxPos.x, qaoaBoxPos.y, qaoaBoxPos.z);
                    rlRotatef(qaoaBoxSpinAngle, 0.f, 1.f, 0.f);
                    DrawCube({0.f, 0.f, 0.f}, qaoaBoxW, qaoaBoxH, qaoaBoxD, boxFill);
                    DrawCubeWires({0.f, 0.f, 0.f}, qaoaBoxW, qaoaBoxH, qaoaBoxD, boxWire);
                    rlPopMatrix();
                }
            }
        }

        EndMode3D();

        if (storyStep == StoryStep::QaoaGateDemo && qMeshN > 0) {
            if (qaoaBoxAlpha > 0.28f && PointInFrontOfCamera(qaoaBoxPos, camera)) {
                Vector2 bs = GetWorldToScreen(qaoaBoxPos, camera);
                if (OnScreen(bs, sw, sh, 140)) {
                    const int fsQ = 22;
                    const char* lab = "QAOA";
                    const int tw = MeasureText(lab, fsQ);
                    const unsigned char ba =
                        static_cast<unsigned char>(std::min(255.f, qaoaBoxAlpha * 255.f));
                    DrawText(lab, static_cast<int>(bs.x) - tw / 2, static_cast<int>(bs.y) - fsQ - 10, fsQ,
                             {200, 230, 255, ba});
                }
            }
            auto drawGateScreen = [&](const Vector3& w, const char* txt, float a) {
                if (a < 0.04f || !PointInFrontOfCamera(w, camera)) {
                    return;
                }
                Vector2 s = GetWorldToScreen(w, camera);
                if (!OnScreen(s, sw, sh, 120)) {
                    return;
                }
                const int fs = (txt[0] == 'Z' && txt[1] == 'Z' && txt[2] == '\0') ? 40 : 48;
                const int tw = MeasureText(txt, fs);
                const unsigned char aa = static_cast<unsigned char>(std::min(255.f, a * 255.f));
                DrawText(txt, static_cast<int>(s.x) - tw / 2, static_cast<int>(s.y) - fs / 2, fs,
                         {248, 252, 255, aa});
            };
            drawGateScreen(qaoaWorldD, "Z", qaoaGateLabelA);
            drawGateScreen(qaoaWorldO, "ZZ", qaoaGateLabelA);
            if (qaoaBundleAlpha > 0.02f) {
                const double maxA = MaxAbsQSubmatrix(surf, qBlockBaseI, qBlockBaseJ, qMeshN);
                const int risePx = static_cast<int>((1.f - qaoaBundleAlpha) * 48.f);
                DrawQaoaBestBundlesMatrix2D(surf, qBlockBaseI, qBlockBaseJ, qMeshN, sw, sh, maxA, qaoaBundleAlpha,
                                          risePx);
            }
        }

        if (storyStep == StoryStep::MarblesMcDecompose) {
            const float flashA = McMatrixLabelAlpha(mcPlayEntrance ? mcElapsed : 1.0e6f);
            if (flashA > 0.02f && mcHaveMatrixLabels) {
                const int fs = 64;
                auto drawMcFlash = [&](const char* ch, Vector3 world, Color rgb) {
                    if (!PointInFrontOfCamera(world, camera)) {
                        return;
                    }
                    Vector2 scr = GetWorldToScreen(world, camera);
                    if (!OnScreen(scr, sw, sh, 80)) {
                        return;
                    }
                    int tw = MeasureText(ch, fs);
                    Color col = rgb;
                    col.a = static_cast<unsigned char>(std::round(flashA * 255.f));
                    DrawText(ch, static_cast<int>(scr.x) - tw / 2, static_cast<int>(scr.y) - fs, fs, col);
                };
                drawMcFlash("M", mcLabelWorldM, {230, 245, 255, 255});
                drawMcFlash("C", mcLabelWorldC, g_style.depArrow);
            }
        }

        DrawStoryMathPanel(storyStep, sw, sh, showLegend, showQMatrix, qLay, surf);

        if (haveQLabelWorld) {
            float qLabA = 0.f;
            if (storyStep == StoryStep::MarblesMcMerge) {
                McMergeStoryPhase qqm = McMergeStoryPhase::FadeOverlays;
                float qmu = 0.f;
                McMergeDecodePhase(mcMergePlayEntrance ? mcMergeElapsed : 1.0e6f, &qqm, &qmu);
                if (static_cast<int>(qqm) < static_cast<int>(McMergeStoryPhase::MergeTeal)) {
                    qLabA = 0.f;
                } else if (qqm == McMergeStoryPhase::MergeTeal) {
                    qLabA = 255.f * SmoothStep((qmu - 0.5f) / 0.5f);
                } else {
                    qLabA = 255.f;
                }
            } else if (storyStep == StoryStep::QSliceToBlock) {
                QSliceStoryPhase qq = QSliceStoryPhase::ShowGrid;
                float qu2 = 0.f;
                QSliceDecodePhase(qSlicePlayEntrance ? qSliceElapsed : 1.0e6f, &qq, &qu2);
                float planeQ = 1.f;
                if (qq == QSliceStoryPhase::Fade2D) {
                    planeQ = 1.f - SmoothStep(qu2);
                } else if (qq == QSliceStoryPhase::Done) {
                    planeQ = 0.f;
                }
                qLabA = 255.f * planeQ;
            }
            if (qLabA > 3.f && PointInFrontOfCamera(qLabelWorld, camera)) {
                Vector2 scrq = GetWorldToScreen(qLabelWorld, camera);
                if (OnScreen(scrq, sw, sh, 90)) {
                    const int qfs = 64;
                    int twq = MeasureText("Q", qfs);
                    Color qcol = {70, 235, 220, static_cast<unsigned char>(qLabA)};
                    DrawText("Q", static_cast<int>(scrq.x) - twq / 2, static_cast<int>(scrq.y) - qfs, qfs, qcol);
                }
            }
        }

        if (storyStep == StoryStep::QBlockField || storyStep == StoryStep::HamiltonianLandscape) {
            const Color diagLabelCol = {120, 220, 255, 255};
            const int labelFs = 12;
            for (int i = 0; i < qMeshN; i++) {
                for (int j = 0; j < qMeshN; j++) {
                    const int gi = qBlockBaseI + i;
                    const int gj = qBlockBaseJ + j;
                    if (gi != gj) {
                        continue;
                    }
                    Vector3 tip = gridPosBlock(i, j);
                    if (!PointInFrontOfCamera(tip, camera)) {
                        continue;
                    }
                    Vector2 scr = GetWorldToScreen(tip, camera);
                    if (!OnScreen(scr, sw, sh, 40)) {
                        continue;
                    }
                    char tag[48];
                    if (gi < surf.nCoverage) {
                        std::snprintf(tag, sizeof(tag), "%d  cov", gi);
                    } else {
                        std::snprintf(tag, sizeof(tag), "%d  slk", gi);
                    }
                    int tw = MeasureText(tag, labelFs);
                    int tx = static_cast<int>(scr.x) - tw / 2;
                    int ty = static_cast<int>(scr.y) - labelFs - 4;
                    DrawText(tag, tx, ty, labelFs, diagLabelCol);
                }
            }
        }

        if (showLegend) {
            DrawLegendPanel(sw, sh, surf, !surf.x.empty(), surf.parametricLambda, surf.lambdaLive, marginScaleS,
                            marginScaleEnabled);
        }

        if (showQMatrix) {
            DrawQMatrixPanel(surf, qLay, scaleDen, sw, sh, mouse);
        }

        if (surf.parametricLambda) {
            if (marginScaleEnabled) {
                DrawText("Margin s (log — drag)   [M] off", static_cast<int>(kLambdaSliderScreenX),
                         static_cast<int>(marginSliderScreenY) - 20, 13, (Color){140, 175, 220, 255});
                DrawMarginScaleSlider(kLambdaSliderScreenX, marginSliderScreenY, marginScaleS, marginSLogMin,
                                      marginSLogMax);
                char mEnds[96];
                std::snprintf(mEnds, sizeof(mEnds), "10^%.2f … 10^%.2f", marginSLogMin, marginSLogMax);
                DrawText(mEnds, static_cast<int>(kLambdaSliderScreenX + kLambdaSliderW + 12),
                         static_cast<int>(marginSliderScreenY + 1.f), 11, (Color){110, 140, 175, 210});
            } else {
                DrawText("Margin s: OFF  —  press M to enable slider", static_cast<int>(kLambdaSliderScreenX),
                         static_cast<int>(marginSliderScreenY) - 6, 13, (Color){95, 105, 125, 220});
            }
            DrawText("Penalty λ (log — drag)", static_cast<int>(kLambdaSliderScreenX),
                     static_cast<int>(lambdaSliderScreenY) - 20, 13, (Color){150, 175, 195, 255});
            DrawLambdaPenaltySlider(kLambdaSliderScreenX, lambdaSliderScreenY, surf.lambdaLive, lambdaLogMin,
                                    lambdaLogMax);
            char ends[120];
            std::snprintf(ends, sizeof(ends), "10^%.2f  …  10^%.2f", lambdaLogMin, lambdaLogMax);
            DrawText(ends, static_cast<int>(kLambdaSliderScreenX + kLambdaSliderW + 12),
                     static_cast<int>(lambdaSliderScreenY + 1.f), 12,
                     (Color){120, 135, 155, 220});
        }
        if (storyStep == StoryStep::HamiltonianLandscape) {
            DrawText("LMB drag: orbit   wheel: zoom   WASD: pan   SPACE: recenter   R: spin   I: legend   Q: matrix   "
                     "M: margin   ← / → : story   B: reset marbles",
                     20, sh - 28, 13, GRAY);
        } else {
            DrawText(
                "LMB drag: orbit   wheel: zoom   WASD: pan   SPACE: recenter   R: spin   I: legend   Q: matrix   "
                "M: margin   ← / → : story",
                20, sh - 28, 13, GRAY);
        }

        EndDrawing();
    }

    ShutdownStoryEquationFont();
    CloseWindow();
    return 0;
}
