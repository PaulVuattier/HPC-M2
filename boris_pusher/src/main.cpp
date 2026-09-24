// Boris pusher test driver: single particles and arrays in uniform fields (SI).
//
// Preset cases (default: writes data/boris/{e_only,b_only,exb,ring}.csv):
//   e_only : E=(1e4,0,0), B=0, v0=0       -> constant acceleration, analytic check
//   b_only : E=0, B=(0,0,1), v0=(1e5,0,0) -> gyromotion, |v|/r/T checks
//   exb    : E=(0,1e4,0), B=(0,0,1), v0=0 -> cycloid, drift v_d = E x B / B^2
//   ring   : 64 protons spread over one gyro-circle -> rigid-rotation array test
//
// Custom single particle (no recompile):
//   ./boris_pusher --run OUT.csv q=.. m=.. x=x,y,z v=x,y,z E=x,y,z B=x,y,z dt=.. steps=..
// Any key may be omitted (defaults: proton q/m, zeros, dt=1e-10, steps=1000).
//
// Gyro-phase ring of N identical particles:
//   ./boris_pusher --run-ring OUT.csv n=128 [q=.. m=.. v0=.. E=.. B=.. dt=.. steps=.. stride=..]
// N particles share one gyro-circle (centre at the origin, radius r) at uniform
// phases; CSV gains a leading pid column: pid,t,x,y,z,vx,vy,vz.
//
// Arbitrary ensemble from an initial-condition file:
//   ./boris_pusher --run-array OUT.csv IC.csv [E=.. B=.. dt=.. steps=.. stride=..]
// IC.csv holds one particle per row, `q,m,x,y,z,vx,vy,vz`, with an optional
// header line. This is the general entry point: heterogeneous charge, mass and
// velocity in a single array (mixed species, thermal spreads, ...).
// `stride=k` writes every k-th step (plus step 0 and the final step) so large
// N x steps runs stay a sane file size.
//
// Array-independence self-test (no output file):
//   ./boris_pusher --selftest-array [n=.. steps=..]
// Pushes one heterogeneous ensemble twice -- once through boris_push_array,
// once particle-by-particle through boris_push -- and demands bitwise equality.
//
// Other modes: ./boris_pusher [OUT_DIR] | ./boris_pusher --stdout <case>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "boris.h"

namespace {
constexpr double Q_PROTON = 1.602176634e-19;    // C
constexpr double M_PROTON = 1.67262192369e-27;  // kg

struct Case {
    const char* name;
    Particle p0;
    Vec3 E;
    Vec3 B;
    double dt;
    int steps;
};

void run_case(const Case& c, FILE* out) {
    Particle p = c.p0;
    std::fprintf(out, "t,x,y,z,vx,vy,vz\n");
    std::fprintf(out, "%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", 0.0, p.x.x,
                 p.x.y, p.x.z, p.v.x, p.v.y, p.v.z);
    for (int n = 1; n <= c.steps; ++n) {
        boris_push(p, c.E, c.B, c.dt);
        double t = n * c.dt;
        std::fprintf(out, "%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", t,
                     p.x.x, p.x.y, p.x.z, p.v.x, p.v.y, p.v.z);
    }
}

bool parse_vec(const char* s, Vec3& v) {
    return std::sscanf(s, "%lf,%lf,%lf", &v.x, &v.y, &v.z) == 3;
}

// ---------------------------------------------------------------------------
// Shared key=value argument helpers
// ---------------------------------------------------------------------------

// Split "key=value"; reports the error itself so callers just `return 1`.
bool split_kv(const char* arg, std::string& k, std::string& v) {
    std::string a = arg;
    auto eq = a.find('=');
    if (eq == std::string::npos) {
        std::fprintf(stderr, "error: bad arg '%s' (want key=value)\n", a.c_str());
        return false;
    }
    k = a.substr(0, eq);
    v = a.substr(eq + 1);
    return true;
}

bool parse_vec_arg(const std::string& k, const std::string& val, Vec3& v) {
    if (parse_vec(val.c_str(), v)) return true;
    std::fprintf(stderr, "error: bad vec '%s=%s' (want a,b,c)\n", k.c_str(),
                 val.c_str());
    return false;
}

// ---------------------------------------------------------------------------
// Array helpers
// ---------------------------------------------------------------------------

// One ensemble snapshot: every particle's state at time t.
// t is printed at full %.12e precision like the state columns: the analysis
// code evaluates the exact solution AT these times, and 7 significant digits
// of t would inject a phase error |omega|*dt_round large enough to swamp the
// real error of a slowly gyrating species.
void write_snapshot(FILE* out, const std::vector<Particle>& ps, double t) {
    const int n = static_cast<int>(ps.size());
    for (int k = 0; k < n; ++k) {
        const Particle& p = ps[static_cast<size_t>(k)];
        std::fprintf(out, "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", k, t,
                     p.x.x, p.x.y, p.x.z, p.v.x, p.v.y, p.v.z);
    }
}

// Advance a whole ensemble and stream it out. Every stride-th step is written,
// and step 0 and the final step always are (so t=0 and t_max are never missed
// however stride divides steps). Mutates ps.
void run_array_case(std::vector<Particle>& ps, const Vec3& E, const Vec3& B,
                    double dt, int steps, int stride, FILE* out) {
    std::fprintf(out, "pid,t,x,y,z,vx,vy,vz\n");
    write_snapshot(out, ps, 0.0);
    for (int s = 1; s <= steps; ++s) {
        boris_push_array(ps, E, B, dt);
        if (s % stride == 0 || s == steps) write_snapshot(out, ps, s * dt);
    }
}

// N identical particles spread uniformly in gyro-phase over ONE gyro-circle of
// radius r = m v0 / (|q| |B|), centred on the origin.
//
// An orthonormal right-handed (u, w, b) basis is built with b = B/|B|, so the
// ring lives in the plane perpendicular to B:
//     x0(ph) =  r * d(ph),          d(ph) = u cos(ph) + w sin(ph)
//     v0(ph) = -sign(q) * v0 * (b x d(ph))
// which puts every particle's gyrocentre exactly on the origin, for either sign
// of q. The ring is therefore a rigid body under the exact dynamics: it spins
// at omega = qB/m without changing shape. (For B = (0,0,1) and q > 0 the basis
// comes out u = (0,1,0), w = (-1,0,0), i.e. x0 = r(-sin ph, cos ph, 0) and
// v = v0(cos ph, sin ph, 0).) Each individual member is a rigid translate of
// the b_only orbit; the ring as a SET is not, since b_only's gyrocentre is
// (0,-r,0) while this ring is centred on the origin.
std::vector<Particle> ring_init(int n, double q, double m, double v0,
                                const Vec3& B) {
    const double Bm = std::sqrt(dot(B, B));
    const double r = m * v0 / (std::fabs(q) * Bm);
    const Vec3 b = B * (1.0 / Bm);
    // Any axis not parallel to b works as the seed for the perpendicular plane.
    Vec3 ax = (std::fabs(b.z) < 0.9) ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
    Vec3 u = ax - b * dot(ax, b);
    u = u * (1.0 / std::sqrt(dot(u, u)));
    const Vec3 w = cross(b, u);
    const double sgn = (q >= 0) ? 1.0 : -1.0;

    std::vector<Particle> ps;
    ps.reserve(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        const double ph = 2.0 * M_PI * k / n;
        const Vec3 d = u * std::cos(ph) + w * std::sin(ph);
        Particle p;
        p.q = q;
        p.m = m;
        p.x = d * r;
        p.v = cross(b, d) * (-sgn * v0);
        ps.push_back(p);
    }
    return ps;
}

// Read an initial-condition CSV: one particle per row, `q,m,x,y,z,vx,vy,vz`.
// A header line (anything not starting with a digit, sign or dot), blank lines
// and '#' comments are skipped.
bool read_ic(const char* path, std::vector<Particle>& ps) {
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "error: cannot open IC file '%s'\n", path);
        return false;
    }
    char line[512];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f)) {
        ++lineno;
        const char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (*s == '\0' || *s == '\n' || *s == '\r' || *s == '#') continue;
        if (!std::isdigit(static_cast<unsigned char>(*s)) && *s != '-' &&
            *s != '+' && *s != '.')
            continue;  // header row
        Particle p;
        const int got =
            std::sscanf(s, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &p.q, &p.m,
                        &p.x.x, &p.x.y, &p.x.z, &p.v.x, &p.v.y, &p.v.z);
        if (got != 8) {
            std::fprintf(stderr,
                         "error: %s:%d: want 8 comma-separated values "
                         "(q,m,x,y,z,vx,vy,vz), got %d\n",
                         path, lineno, got);
            std::fclose(f);
            return false;
        }
        if (!(p.m > 0.0)) {
            std::fprintf(stderr, "error: %s:%d: m must be > 0\n", path, lineno);
            std::fclose(f);
            return false;
        }
        ps.push_back(p);
    }
    std::fclose(f);
    if (ps.empty()) {
        std::fprintf(stderr, "error: no particles found in '%s'\n", path);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

// Generic single run: --run OUT.csv [q=.. m=.. x=.. v=.. E=.. B=.. dt=.. steps=..]
int run_custom(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "error: --run needs an output path\n");
        return 1;
    }
    Case c = {"custom", Particle{}, {0, 0, 0}, {0, 0, 0}, 1.0e-10, 1000};
    c.p0.q = Q_PROTON;
    c.p0.m = M_PROTON;
    for (int i = 3; i < argc; ++i) {
        std::string k, val;
        if (!split_kv(argv[i], k, val)) return 1;
        char* end = nullptr;
        if (k == "q")
            c.p0.q = std::strtod(val.c_str(), &end);
        else if (k == "m")
            c.p0.m = std::strtod(val.c_str(), &end);
        else if (k == "dt")
            c.dt = std::strtod(val.c_str(), &end);
        else if (k == "steps")
            c.steps = std::atoi(val.c_str());
        else if (k == "x") {
            if (!parse_vec_arg(k, val, c.p0.x)) return 1;
        } else if (k == "v") {
            if (!parse_vec_arg(k, val, c.p0.v)) return 1;
        } else if (k == "E") {
            if (!parse_vec_arg(k, val, c.E)) return 1;
        } else if (k == "B") {
            if (!parse_vec_arg(k, val, c.B)) return 1;
        } else {
            std::fprintf(stderr, "error: unknown key '%s'\n", k.c_str());
            return 1;
        }
        if (end && *end != '\0') {
            std::fprintf(stderr, "error: bad value '%s=%s'\n", k.c_str(),
                         val.c_str());
            return 1;
        }
    }
    if (c.steps < 0 || c.dt <= 0 || c.p0.m <= 0) {
        std::fprintf(stderr, "error: need steps>=0, dt>0, m>0\n");
        return 1;
    }
    FILE* out = std::fopen(argv[2], "w");
    if (!out) {
        std::fprintf(stderr, "error: cannot open '%s' for writing\n", argv[2]);
        return 1;
    }
    run_case(c, out);
    std::fclose(out);
    std::fprintf(stderr, "wrote %s (%d steps, dt=%.3e s)\n", argv[2], c.steps,
                 c.dt);
    return 0;
}

// Gyro-phase ring: --run-ring OUT.csv [n=.. q=.. m=.. v0=.. E=.. B=.. dt=.. steps=.. stride=..]
int run_ring(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "error: --run-ring needs an output path\n");
        return 1;
    }
    double q = Q_PROTON, m = M_PROTON, v0 = 1.0e5, dt = 1.0e-10;
    int n = 128, steps = 2000, stride = 1;
    Vec3 E = {0, 0, 0}, B = {0, 0, 1.0};
    for (int i = 3; i < argc; ++i) {
        std::string k, val;
        if (!split_kv(argv[i], k, val)) return 1;
        char* end = nullptr;
        if (k == "q")
            q = std::strtod(val.c_str(), &end);
        else if (k == "m")
            m = std::strtod(val.c_str(), &end);
        else if (k == "v0")
            v0 = std::strtod(val.c_str(), &end);
        else if (k == "dt")
            dt = std::strtod(val.c_str(), &end);
        else if (k == "n")
            n = std::atoi(val.c_str());
        else if (k == "steps")
            steps = std::atoi(val.c_str());
        else if (k == "stride")
            stride = std::atoi(val.c_str());
        else if (k == "E") {
            if (!parse_vec_arg(k, val, E)) return 1;
        } else if (k == "B") {
            if (!parse_vec_arg(k, val, B)) return 1;
        } else {
            std::fprintf(stderr, "error: unknown key '%s'\n", k.c_str());
            return 1;
        }
        if (end && *end != '\0') {
            std::fprintf(stderr, "error: bad value '%s=%s'\n", k.c_str(),
                         val.c_str());
            return 1;
        }
    }
    if (n < 1 || steps < 0 || stride < 1 || dt <= 0 || m <= 0 || v0 < 0 ||
        q == 0.0 || dot(B, B) == 0.0) {
        std::fprintf(
            stderr,
            "error: need n>=1, steps>=0, stride>=1, dt>0, m>0, v0>=0, q!=0, B!=0\n");
        return 1;
    }
    std::vector<Particle> ps = ring_init(n, q, m, v0, B);
    FILE* out = std::fopen(argv[2], "w");
    if (!out) {
        std::fprintf(stderr, "error: cannot open '%s' for writing\n", argv[2]);
        return 1;
    }
    run_array_case(ps, E, B, dt, steps, stride, out);
    std::fclose(out);
    std::fprintf(stderr,
                 "wrote %s (ring n=%d, %d steps, stride %d, dt=%.3e s)\n",
                 argv[2], n, steps, stride, dt);
    return 0;
}

// Arbitrary ensemble: --run-array OUT.csv IC.csv [E=.. B=.. dt=.. steps=.. stride=..]
int run_array(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "error: --run-array needs an output path and an IC file\n");
        return 1;
    }
    double dt = 1.0e-10;
    int steps = 1000, stride = 1;
    Vec3 E = {0, 0, 0}, B = {0, 0, 0};
    for (int i = 4; i < argc; ++i) {
        std::string k, val;
        if (!split_kv(argv[i], k, val)) return 1;
        char* end = nullptr;
        if (k == "dt")
            dt = std::strtod(val.c_str(), &end);
        else if (k == "steps")
            steps = std::atoi(val.c_str());
        else if (k == "stride")
            stride = std::atoi(val.c_str());
        else if (k == "E") {
            if (!parse_vec_arg(k, val, E)) return 1;
        } else if (k == "B") {
            if (!parse_vec_arg(k, val, B)) return 1;
        } else {
            std::fprintf(stderr, "error: unknown key '%s'\n", k.c_str());
            return 1;
        }
        if (end && *end != '\0') {
            std::fprintf(stderr, "error: bad value '%s=%s'\n", k.c_str(),
                         val.c_str());
            return 1;
        }
    }
    if (steps < 0 || stride < 1 || dt <= 0) {
        std::fprintf(stderr, "error: need steps>=0, stride>=1, dt>0\n");
        return 1;
    }
    std::vector<Particle> ps;
    if (!read_ic(argv[3], ps)) return 1;
    FILE* out = std::fopen(argv[2], "w");
    if (!out) {
        std::fprintf(stderr, "error: cannot open '%s' for writing\n", argv[2]);
        return 1;
    }
    run_array_case(ps, E, B, dt, steps, stride, out);
    std::fclose(out);
    std::fprintf(stderr,
                 "wrote %s (array n=%zu from %s, %d steps, stride %d, "
                 "dt=%.3e s)\n",
                 argv[2], ps.size(), argv[3], steps, stride, dt);
    return 0;
}

// --selftest-array [n=.. steps=..]
//
// boris_push_array must be *exactly* N independent boris_push calls: no shared
// state, no aliasing, no neighbour's velocity leaking in. Today the array
// function is a plain serial loop, so equality holds by construction -- the
// point of this test is to be the regression guard that fails the moment that
// loop is rewritten (OpenMP, SIMD, GPU) in a way that breaks independence.
// Equality is demanded bitwise: identical inputs through identical arithmetic
// must give identical doubles, so anything but an exact match is a bug, not a
// tolerance question.
int selftest_array(int argc, char* argv[]) {
    int n = 512, steps = 200;
    for (int i = 2; i < argc; ++i) {
        std::string k, val;
        if (!split_kv(argv[i], k, val)) return 1;
        if (k == "n")
            n = std::atoi(val.c_str());
        else if (k == "steps")
            steps = std::atoi(val.c_str());
        else {
            std::fprintf(stderr, "error: unknown key '%s'\n", k.c_str());
            return 1;
        }
    }
    if (n < 1 || steps < 0) {
        std::fprintf(stderr, "error: need n>=1, steps>=0\n");
        return 1;
    }

    // Deterministic heterogeneous ensemble: mixed charge signs, masses spanning
    // a decade, random positions and velocities.
    std::mt19937 rng(12345u);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<Particle> ps;
    ps.reserve(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        // One draw per statement: operands of * are unsequenced in C++17, so
        // folding several U(rng) calls into one expression would make the
        // "deterministic" ensemble depend on the compiler's evaluation order.
        const double sign = U(rng) < 0.0 ? -1.0 : 1.0;
        const double qscale = 1.0 + 3.0 * std::fabs(U(rng));
        const double mscale = 0.5 + 9.5 * std::fabs(U(rng));
        const double x0 = U(rng), x1 = U(rng), x2 = U(rng);
        const double v0 = U(rng), v1 = U(rng), v2 = U(rng);
        Particle p;
        p.q = Q_PROTON * sign * qscale;
        p.m = M_PROTON * mscale;
        p.x = {x0 * 1.0e-3, x1 * 1.0e-3, x2 * 1.0e-3};
        p.v = {v0 * 1.0e5, v1 * 1.0e5, v2 * 1.0e5};
        ps.push_back(p);
    }

    const Vec3 E = {1.0e3, -2.0e3, 5.0e2};
    const Vec3 B = {0.1, -0.2, 1.0};
    const double dt = 1.0e-11;

    std::vector<Particle> arr = ps;  // reference: one boris_push_array call
    std::vector<Particle> fwd = ps;  // singles, ascending index
    std::vector<Particle> rev = ps;  // singles, descending index
    std::vector<Particle> lo(ps.begin(), ps.begin() + n / 2);  // chunk A
    std::vector<Particle> hi(ps.begin() + n / 2, ps.end());    // chunk B
    for (int s = 0; s < steps; ++s) {
        boris_push_array(arr, E, B, dt);
        for (int k = 0; k < n; ++k)
            boris_push(fwd[static_cast<size_t>(k)], E, B, dt);
        for (int k = n - 1; k >= 0; --k)
            boris_push(rev[static_cast<size_t>(k)], E, B, dt);
        boris_push_array(lo, E, B, dt);
        boris_push_array(hi, E, B, dt);
    }
    std::vector<Particle> chunk = lo;
    chunk.insert(chunk.end(), hi.begin(), hi.end());

    // Three independent ways of covering the same N pushes. Forward singles
    // catch cross-particle coupling; reverse order catches a loop-carried
    // dependency that a same-direction comparison would hide; splitting the
    // array in two mimics an OpenMP domain decomposition.
    const char* label[3] = {"vs singles (fwd)", "vs singles (rev)",
                            "vs 2-chunk split"};
    const std::vector<Particle>* other[3] = {&fwd, &rev, &chunk};
    int bad_total = 0;
    for (int c = 0; c < 3; ++c) {
        int bad = 0;
        for (int k = 0; k < n; ++k) {
            const Particle& a = arr[static_cast<size_t>(k)];
            const Particle& b = (*other[c])[static_cast<size_t>(k)];
            const bool same = a.x.x == b.x.x && a.x.y == b.x.y &&
                              a.x.z == b.x.z && a.v.x == b.v.x &&
                              a.v.y == b.v.y && a.v.z == b.v.z;
            if (!same) ++bad;
        }
        bad_total += bad;
        std::printf("[%s] selftest-array %s: %d/%d particles bitwise identical\n",
                    bad ? "FAIL" : "PASS", label[c], n - bad, n);
    }
    std::printf("       (n=%d, %d steps, tilted E and B, mixed q and m)\n", n,
                steps);
    return bad_total ? 1 : 0;
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--run") return run_custom(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--run-ring")
        return run_ring(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--run-array")
        return run_array(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--selftest-array")
        return selftest_array(argc, argv);
    std::string out_dir = "data/boris";
    bool to_stdout = false;
    std::string stdout_case = "b_only";
    if (argc > 1) {
        std::string a1 = argv[1];
        if (a1 == "--stdout") {
            to_stdout = true;
            if (argc > 2) stdout_case = argv[2];
        } else {
            out_dir = a1;
        }
    }

    Particle base;
    base.q = Q_PROTON;
    base.m = M_PROTON;

    // E-only: start at rest, constant acceleration along x.
    Case e_only = {"e_only",
                   base,
                   {1.0e4, 0.0, 0.0},  // E
                   {0.0, 0.0, 0.0},    // B
                   1.0e-10,
                   1000};
    e_only.p0.x = {0.0, 0.0, 0.0};
    e_only.p0.v = {0.0, 0.0, 0.0};

    // B-only: gyro-orbit, ~3 periods (T ~ 65.6 ns, dt = 0.1 ns).
    Case b_only = {"b_only",
                   base,
                   {0.0, 0.0, 0.0},  // E
                   {0.0, 0.0, 1.0},  // B
                   1.0e-10,
                   2000};
    b_only.p0.x = {0.0, 0.0, 0.0};
    b_only.p0.v = {1.0e5, 0.0, 0.0};

    // ExB: start at rest -> cycloid, drift along +x, ~10 periods.
    Case exb = {"exb",
                base,
                {0.0, 1.0e4, 0.0},  // E
                {0.0, 0.0, 1.0},    // B
                1.0e-10,
                7000};
    exb.p0.x = {0.0, 0.0, 0.0};
    exb.p0.v = {0.0, 0.0, 0.0};

    const Case cases[3] = {e_only, b_only, exb};

    if (to_stdout) {
        for (const auto& c : cases) {
            if (stdout_case == c.name) {
                run_case(c, stdout);
                return 0;
            }
        }
        std::fprintf(stderr,
                     "error: unknown case '%s' "
                     "(e_only|b_only|exb; ring is file-only)\n",
                     stdout_case.c_str());
        return 1;
    }

    for (const auto& c : cases) {
        std::string path = out_dir + "/" + c.name + ".csv";
        FILE* out = std::fopen(path.c_str(), "w");
        if (!out) {
            std::fprintf(stderr, "error: cannot open '%s' for writing\n",
                         path.c_str());
            return 1;
        }
        run_case(c, out);
        std::fclose(out);
        std::fprintf(stderr, "wrote %s (%d steps, dt=%.3e s)\n", path.c_str(),
                     c.steps, c.dt);
    }

    // ring: the array preset. 64 protons over one gyro-circle, ~3 periods.
    // stride=10 keeps the file ~1 MB instead of ~12 MB.
    {
        const int n = 64, steps = 2000, stride = 10;
        const double dt = 1.0e-10, v0 = 1.0e5;
        const Vec3 E = {0.0, 0.0, 0.0}, B = {0.0, 0.0, 1.0};
        std::vector<Particle> ps = ring_init(n, Q_PROTON, M_PROTON, v0, B);
        std::string path = out_dir + "/ring.csv";
        FILE* out = std::fopen(path.c_str(), "w");
        if (!out) {
            std::fprintf(stderr, "error: cannot open '%s' for writing\n",
                         path.c_str());
            return 1;
        }
        run_array_case(ps, E, B, dt, steps, stride, out);
        std::fclose(out);
        std::fprintf(stderr,
                     "wrote %s (ring n=%d, %d steps, stride %d, dt=%.3e s)\n",
                     path.c_str(), n, steps, stride, dt);
    }
    return 0;
}
