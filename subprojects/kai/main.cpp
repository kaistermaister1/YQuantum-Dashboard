// Kai QAOA grid viewer using the exact Raylib camera framework style from qubo_vis.

#include <raylib.h>
#include <raymath.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct GridFrame {
    int iteration = 0;
    int packageIndex = -1;
    double bestObjective = 0.0;
    double bestQuboEnergy = 0.0;
    double currentProfit = 0.0;
    bool improved = false;
    std::string note;
    std::vector<int> matrix;
    std::vector<int> changed;
};

struct GridHistory {
    int nCoverages = 0;
    int nPackages = 0;
    std::vector<std::string> coverageNames;
    std::vector<std::string> packageNames;
    std::vector<double> coeffs;
    std::vector<GridFrame> frames;
};

static bool ReadContentLine(std::ifstream& f, std::string& line) {
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        return true;
    }
    return false;
}

static bool ReadExpectedTag(std::ifstream& f, const char* tag, std::string& err) {
    std::string line;
    if (!ReadContentLine(f, line)) {
        err = std::string("missing tag: ") + tag;
        return false;
    }
    if (line != tag) {
        err = std::string("expected tag '") + tag + "' but found '" + line + "'";
        return false;
    }
    return true;
}

static bool ParseOptionalDoubleToken(const std::string& token, double* out) {
    const std::string lowered = [&token]() {
        std::string value = token;
        for (char& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return value;
    }();
    if (lowered == "nan" || lowered == "none" || lowered == "--") {
        *out = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    char* endPtr = nullptr;
    const double value = std::strtod(token.c_str(), &endPtr);
    if (endPtr == token.c_str() || (endPtr != nullptr && *endPtr != '\0')) {
        return false;
    }
    *out = value;
    return true;
}

static bool ReadIntGrid(std::ifstream& f, int rows, int cols, std::vector<int>& out, std::string& err) {
    out.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
    for (int r = 0; r < rows; r++) {
        std::string line;
        if (!ReadContentLine(f, line)) {
            err = "unexpected EOF reading integer grid";
            return false;
        }
        std::istringstream iss(line);
        for (int c = 0; c < cols; c++) {
            int value = 0;
            if (!(iss >> value)) {
                err = "short integer grid row";
                return false;
            }
            out[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = value ? 1 : 0;
        }
    }
    return true;
}

static bool LoadGridHistory(const char* path, GridHistory& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = std::string("cannot open ") + path;
        return false;
    }

    std::string line;
    if (!ReadContentLine(f, line)) {
        err = "empty file";
        return false;
    }
    if (line != "grid_history_v1") {
        err = "unsupported file header";
        return false;
    }

    if (!ReadContentLine(f, line)) {
        err = "missing dimensions line";
        return false;
    }
    {
        std::istringstream iss(line);
        int frameCount = 0;
        if (!(iss >> out.nCoverages >> out.nPackages >> frameCount)) {
            err = "bad dimensions line";
            return false;
        }
        if (out.nCoverages <= 0 || out.nPackages <= 0 || frameCount <= 0) {
            err = "dimensions must be positive";
            return false;
        }
        out.frames.reserve(static_cast<size_t>(frameCount));
    }

    out.coverageNames.clear();
    out.packageNames.clear();
    out.coeffs.assign(static_cast<size_t>(out.nCoverages) * static_cast<size_t>(out.nPackages), 0.0);

    for (int i = 0; i < out.nCoverages; i++) {
        if (!ReadContentLine(f, line)) {
            err = "missing coverage name";
            return false;
        }
        const std::string prefix = "COVERAGE\t";
        if (line.rfind(prefix, 0) != 0) {
            err = "bad coverage line";
            return false;
        }
        out.coverageNames.push_back(line.substr(prefix.size()));
    }

    for (int i = 0; i < out.nPackages; i++) {
        if (!ReadContentLine(f, line)) {
            err = "missing package name";
            return false;
        }
        const std::string prefix = "PACKAGE\t";
        if (line.rfind(prefix, 0) != 0) {
            err = "bad package line";
            return false;
        }
        out.packageNames.push_back(line.substr(prefix.size()));
    }

    if (!ReadExpectedTag(f, "COEFFS", err)) {
        return false;
    }
    for (int r = 0; r < out.nCoverages; r++) {
        if (!ReadContentLine(f, line)) {
            err = "missing coefficient row";
            return false;
        }
        std::istringstream iss(line);
        for (int c = 0; c < out.nPackages; c++) {
            double value = 0.0;
            if (!(iss >> value)) {
                err = "short coefficient row";
                return false;
            }
            out.coeffs[static_cast<size_t>(r) * static_cast<size_t>(out.nPackages) + static_cast<size_t>(c)] = value;
        }
    }

    while (ReadContentLine(f, line)) {
        if (line.rfind("FRAME ", 0) != 0) {
            err = "expected FRAME line";
            return false;
        }
        GridFrame frame;
        {
            std::istringstream iss(line.substr(6));
            std::string bestObjectiveToken;
            std::string bestQuboToken;
            int improvedInt = 0;
            if (!(iss >> frame.iteration >> frame.packageIndex >> bestObjectiveToken >> bestQuboToken >>
                  frame.currentProfit >> improvedInt)) {
                err = "bad FRAME line";
                return false;
            }
            if (!ParseOptionalDoubleToken(bestObjectiveToken, &frame.bestObjective) ||
                !ParseOptionalDoubleToken(bestQuboToken, &frame.bestQuboEnergy)) {
                err = "bad FRAME line";
                return false;
            }
            frame.improved = improvedInt != 0;
        }

        if (!ReadContentLine(f, line)) {
            err = "missing NOTE line";
            return false;
        }
        const std::string notePrefix = "NOTE\t";
        if (line.rfind(notePrefix, 0) != 0) {
            err = "bad NOTE line";
            return false;
        }
        frame.note = line.substr(notePrefix.size());

        if (!ReadIntGrid(f, out.nCoverages, out.nPackages, frame.matrix, err)) {
            return false;
        }
        if (!ReadExpectedTag(f, "CHANGE", err)) {
            return false;
        }
        if (!ReadIntGrid(f, out.nCoverages, out.nPackages, frame.changed, err)) {
            return false;
        }
        out.frames.push_back(frame);
    }

    if (out.frames.empty()) {
        err = "history contains no frames";
        return false;
    }
    return true;
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

static int FrameValue(const GridFrame& frame, int row, int col, int nPackages) {
    return frame.matrix[static_cast<size_t>(row) * static_cast<size_t>(nPackages) + static_cast<size_t>(col)];
}

static constexpr int kMatrixPadding = 1;
static constexpr double kPlaybackSecondsPerFrame = 0.15;

int main(int argc, char** argv) {
    const char* path = (argc >= 2) ? argv[1] : "qaoa_grid_history.txt";
    GridHistory history;
    std::string err;
    if (!LoadGridHistory(path, history, err)) {
        std::fprintf(stderr, "Load error: %s\n", err.c_str());
        return 1;
    }

    const int sw = 1400;
    const int sh = 900;
    InitWindow(sw, sh, "QAOA Grid Evolution");
    SetTargetFPS(120);

    const int latticeRows = history.nCoverages + 2 * kMatrixPadding;
    const int latticeCols = history.nPackages + 2 * kMatrixPadding;
    const float span = static_cast<float>(std::max(latticeRows, latticeCols));
    const float gx = 1.35f;
    const float gz = 1.0f;
    const float ox = -(static_cast<float>(latticeCols) - 1.0f) * gx * 0.5f;
    const float oz = -(static_cast<float>(latticeRows) - 1.0f) * gz * 0.5f;
    const float amp = 1.0f;

    Camera3D camera{};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.position = {span * 0.9f, span * 0.55f, span * 0.9f};
    camera.up = {0.0f, 1.0f, 0.0f};

    auto latticePoint = [&](int latticeRow, int latticeCol, float height) -> Vector3 {
        return {ox + static_cast<float>(latticeCol) * gx, height, oz + static_cast<float>(latticeRow) * gz};
    };

    auto latticeHeight = [&](const GridFrame& frame, int latticeRow, int latticeCol) -> float {
        const int row = latticeRow - kMatrixPadding;
        const int col = latticeCol - kMatrixPadding;
        if (row < 0 || row >= history.nCoverages || col < 0 || col >= history.nPackages) {
            return 0.0f;
        }
        return FrameValue(frame, row, col, history.nPackages) ? amp : 0.0f;
    };

    bool autoSpin = false;
    bool autoplay = true;
    double frameAccumulator = 0.0;
    int frameIndex = 0;

    const Color meshColor = {245, 245, 247, 255};

    while (!WindowShouldClose()) {
        StepTrackball(camera);
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) autoSpin = false;
        if (IsKeyPressed(KEY_R)) autoSpin = !autoSpin;
        if (IsKeyPressed(KEY_P)) autoplay = !autoplay;
        if (IsKeyPressed(KEY_HOME)) {
            frameIndex = 0;
            frameAccumulator = 0.0;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            frameIndex = std::min(frameIndex + 1, static_cast<int>(history.frames.size()) - 1);
            frameAccumulator = 0.0;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            frameIndex = std::max(frameIndex - 1, 0);
            frameAccumulator = 0.0;
        }

        if (autoplay && frameIndex + 1 < static_cast<int>(history.frames.size())) {
            frameAccumulator += GetFrameTime();
            if (frameAccumulator >= kPlaybackSecondsPerFrame) {
                frameAccumulator = 0.0;
                frameIndex += 1;
            }
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

        const GridFrame& frame = history.frames[static_cast<size_t>(frameIndex)];

        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode3D(camera);

        for (int row = 0; row < latticeRows; row++) {
            for (int col = 0; col < latticeCols; col++) {
                const Vector3 here = latticePoint(row, col, latticeHeight(frame, row, col));
                if (col + 1 < latticeCols) {
                    DrawLine3D(here, latticePoint(row, col + 1, latticeHeight(frame, row, col + 1)), meshColor);
                }
                if (row + 1 < latticeRows) {
                    DrawLine3D(here, latticePoint(row + 1, col, latticeHeight(frame, row + 1, col)), meshColor);
                }
            }
        }

        EndMode3D();

        DrawText("QAOA GRID EVOLUTION", 20, 18, 18, RAYWHITE);
        DrawText("Single white padded-lattice mesh with binary heights", 20, 42, 15,
                 (Color){180, 190, 205, 255});

        DrawRectangle(18, sh - 132, sw - 36, 94, (Color){8, 10, 14, 230});
        DrawRectangleLines(18, sh - 132, sw - 36, 94, (Color){55, 70, 90, 255});

        char hudA[512];
        char hudB[512];
        char hudC[512];
        const char* packageLabel =
            (frame.packageIndex >= 0 && frame.packageIndex < static_cast<int>(history.packageNames.size()))
                ? history.packageNames[static_cast<size_t>(frame.packageIndex)].c_str()
                : "setup";
        std::snprintf(hudA, sizeof(hudA), "Iteration %d / %d   |   current best package: %s   |   improved: %s",
                      frame.iteration, std::max(0, static_cast<int>(history.frames.size()) - 1), packageLabel,
                      frame.improved ? "yes" : "no");
        std::snprintf(hudB, sizeof(hudB),
                      "Binary height field: x(i,m) in {0,1} on interior intersections   |   Current selected profit: %.6g",
                      frame.currentProfit);
        std::snprintf(hudC, sizeof(hudC),
                      "LMB: rotate   wheel: zoom   WASD: pan   SPACE: recenter   P: play/pause   <- / -> : step   HOME: reset   R: spin   autoplay=%.2fs/frame",
                      kPlaybackSecondsPerFrame);
        DrawText(hudA, 30, sh - 118, 17, (Color){210, 220, 235, 255});
        DrawText(hudB, 30, sh - 90, 16, (Color){120, 200, 255, 255});
        DrawText(frame.note.c_str(), 30, sh - 64, 15, (Color){227, 24, 55, 255});
        DrawText(hudC, 20, sh - 28, 15, GRAY);

        int rightY = 84;
        DrawText("Packages", sw - 220, rightY, 15, RAYWHITE);
        rightY += 22;
        for (int c = 0; c < history.nPackages; c++) {
            DrawText(history.packageNames[static_cast<size_t>(c)].c_str(), sw - 220, rightY, 12,
                     c == frame.packageIndex ? meshColor : (Color){180, 190, 205, 255});
            rightY += 16;
        }

        DrawText(path, 20, 64, 14, GRAY);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
