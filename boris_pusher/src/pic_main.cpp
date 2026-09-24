// pic_pusher: PIC Boris driver (1D/2D selectable, full PIC loop).
//
// Modes:
//   ./pic_pusher --run-pic OUT.csv IC.csv --dim 1|2 [options]
//     Full PIC loop: gather -> Boris move -> deposit -> Ohm/Faraday.
//     IC.csv rows: q,m,x,y,z,vx,vy,vz[,w] (header/#-comments skipped,
//     w = macro weight, default 1). Output: pid,t,x,y,z,vx,vy,vz.
//   ./pic_pusher --selftest-pic [--dim 1|2]
//     In-memory guards: weight conservation (P2), uniform gather
//     exactness (P3), divB conservation (P5), uniform PIC-vs-analytic
//     equivalence (P1). No output file.
//
// Common options (key=value):
//   --dim 1|2  --nx N --ny M --dx .. --dy .. --dt .. --steps N
//   --stride k --te ..(J) --field uniform|sine --E x,y,z --B x,y,z
//   --freeze none|E|B|EB  (hold fields fixed: prescribed-field tests)
//   --diag FILE.csv      (energy / particle-number time series every stride)
//   --q0 .. --m0 ..       (fallback q/m when IC gives q=m=0, e.g. neutral
//                          placeholder rows; default proton)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "boris.h"
#include "pic.h"

namespace {

constexpr double Q_PROTON = 1.602176634e-19;
constexpr double M_PROTON = 1.67262192369e-27;

struct Cfg {
    int dim = 1;
    int nx = 64, ny = 64;
    double dx = 1.0e-4, dy = 1.0e-4;
    double dt = 1.0e-10;
    int steps = 1000, stride = 10;
    double kTe = 0.0;
    std::string field = "uniform";
    Vec3 E0{0, 0, 0}, B0{0, 0, 1.0};
    bool freeze_E = false, freeze_B = false;
    double q0 = Q_PROTON, m0 = M_PROTON;
    std::string grid_path; // optional: dump final E,B,n,vi grid to CSV
    std::string diag_path; // optional: conservation time series (see write_diag)
};

bool split_kv(const char* arg, std::string& k, std::string& v) {
    std::string a = arg;
    auto eq = a.find('=');
    if (eq == std::string::npos) {
        std::fprintf(stderr, "error: bad arg '%s' (want key=value)\n",
                     a.c_str());
        return false;
    }
    k = a.substr(0, eq);
    v = a.substr(eq + 1);
    return true;
}

bool parse_vec(const std::string& s, Vec3& v) {
    return std::sscanf(s.c_str(), "%lf,%lf,%lf", &v.x, &v.y, &v.z) == 3;
}

bool parse_cfg(int argc, char* argv[], int from, Cfg& c) {
    for (int i = from; i < argc; ++i) {
        std::string k, v;
        if (!split_kv(argv[i], k, v)) return false;
        char* end = nullptr;
        if (k == "dim")
            c.dim = std::atoi(v.c_str());
        else if (k == "nx")
            c.nx = std::atoi(v.c_str());
        else if (k == "ny")
            c.ny = std::atoi(v.c_str());
        else if (k == "dx")
            c.dx = std::strtod(v.c_str(), &end);
        else if (k == "dy")
            c.dy = std::strtod(v.c_str(), &end);
        else if (k == "dt")
            c.dt = std::strtod(v.c_str(), &end);
        else if (k == "steps")
            c.steps = std::atoi(v.c_str());
        else if (k == "stride")
            c.stride = std::atoi(v.c_str());
        else if (k == "te")
            c.kTe = std::strtod(v.c_str(), &end);
        else if (k == "field")
            c.field = v;
        else if (k == "freeze") {
            c.freeze_E = (v == "E" || v == "EB");
            c.freeze_B = (v == "B" || v == "EB");
            if (v != "none" && !c.freeze_E && !c.freeze_B) {
                std::fprintf(stderr, "error: freeze=%s (want none|E|B|EB)\n",
                             v.c_str());
                return false;
            }
        } else if (k == "E") {
            if (!parse_vec(v, c.E0)) return false;
        } else if (k == "B") {
            if (!parse_vec(v, c.B0)) return false;
        }         else if (k == "q0")
            c.q0 = std::strtod(v.c_str(), &end);
        else if (k == "m0")
            c.m0 = std::strtod(v.c_str(), &end);
        else if (k == "grid")
            c.grid_path = v;
        else if (k == "diag")
            c.diag_path = v;
        else {
            std::fprintf(stderr, "error: unknown key '%s'\n", k.c_str());
            return false;
        }
        if (end && *end != '\0') {
            std::fprintf(stderr, "error: bad value '%s=%s'\n", k.c_str(),
                         v.c_str());
            return false;
        }
    }
    if ((c.dim != 1 && c.dim != 2) || c.nx < 2 || c.ny < 2 || c.dx <= 0 ||
        c.dy <= 0 || c.dt <= 0 || c.steps < 0 || c.stride < 1) {
        std::fprintf(stderr, "error: bad grid/time params\n");
        return false;
    }
    return true;
}

bool read_ic(const char* path, std::vector<pic::MacroParticle>& ps, double q0,
             double m0) {
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "error: cannot open IC file '%s'\n", path);
        return false;
    }
    char line[1024];
    int lineno = 0;
    while (std::fgets(line, sizeof line, f)) {
        ++lineno;
        const char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (*s == '\0' || *s == '\n' || *s == '\r' || *s == '#') continue;
        if ((*s < '0' || *s > '9') && *s != '-' && *s != '+' && *s != '.')
            continue; // header
        pic::MacroParticle mp;
        double q, m, x, y, z, vx, vy, vz, w = 1.0;
        int got =
            std::sscanf(s, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &q, &m, &x,
                        &y, &z, &vx, &vy, &vz, &w);
        if (got != 8 && got != 9) {
            std::fprintf(stderr, "error: %s:%d: want 8-9 values, got %d\n",
                         path, lineno, got);
            std::fclose(f);
            return false;
        }
        if (m <= 0) {
            if (m == 0) {
                m = m0;
                q = (q == 0) ? q0 : q;
            } else {
                std::fprintf(stderr, "error: %s:%d: m must be > 0\n", path,
                             lineno);
                std::fclose(f);
                return false;
            }
        }
        mp.p.q = q;
        mp.p.m = m;
        mp.p.x = {x, y, z};
        mp.p.v = {vx, vy, vz};
        mp.w = w;
        ps.push_back(mp);
    }
    std::fclose(f);
    if (ps.empty()) {
        std::fprintf(stderr, "error: no particles in '%s'\n", path);
        return false;
    }
    return true;
}

void write_snapshot(FILE* out, const std::vector<pic::MacroParticle>& ps,
                    double t) {
    for (size_t k = 0; k < ps.size(); ++k) {
        const Particle& p = ps[k].p;
        std::fprintf(out, "%zu,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", k,
                     t, p.x.x, p.x.y, p.x.z, p.v.x, p.v.y, p.v.z);
    }
}

void write_grid_1d(FILE* out, const pic::Grid1D& g,
                   const std::vector<double>& n, const std::vector<double>& vx,
                   const std::vector<double>& vy,
                   const std::vector<double>& vz) {
    std::fprintf(out, "i,x,Ex,Ey,Ez,Bx,By,Bz,n,vx,vy,vz\n");
    for (int i = 0; i < g.nx; ++i) {
        const size_t k = (size_t)i;
        std::fprintf(out, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,"
                          "%.17e,%.17e,%.17e\n",
                     i, g.x0 + i * g.dx, g.E[0][k], g.E[1][k], g.E[2][k],
                     g.B[0][k], g.B[1][k], g.B[2][k],
                     k < n.size() ? n[k] : 0.0, k < vx.size() ? vx[k] : 0.0,
                     k < vy.size() ? vy[k] : 0.0, k < vz.size() ? vz[k] : 0.0);
    }
}

void write_grid_2d(FILE* out, const pic::Grid2D& g,
                   const std::vector<double>& n, const std::vector<double>& vx,
                   const std::vector<double>& vy,
                   const std::vector<double>& vz) {
    std::fprintf(out, "i,j,x,y,Ex,Ey,Ez,Bx,By,Bz,n,vx,vy,vz\n");
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const size_t c = g.id(i, j);
            std::fprintf(out, "%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,"
                              "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                         i, j, g.x0 + i * g.dx, g.y0 + j * g.dy, g.E[0][c],
                         g.E[1][c], g.E[2][c], g.B[0][c], g.B[1][c],
                         g.B[2][c], c < n.size() ? n[c] : 0.0,
                         c < vx.size() ? vx[c] : 0.0,
                         c < vy.size() ? vy[c] : 0.0,
                         c < vz.size() ? vz[c] : 0.0);
        }
}

// Conservation diagnostics, one row per call (every `stride` steps):
//   K     = sum 1/2 m w |v|^2                    ion kinetic energy
//   WdB   = sum |B - <B>|^2 / (2 mu0) dV         perturbation magnetic energy
//           (<B> is conserved by periodic Faraday, so the cross term with the
//           background vanishes and K + WdB is the energy that must stay put)
//   Ndep  = sum n dV                             deposited particle number
//   Wsum  = sum w                                what Ndep must equal
//   Bz_mean, max_divB (0 in 1D: Bx is static)
//   Px,Py,Pz = sum m w v                         total ion momentum
// Energies are per unit transverse area (1D) or length (2D).
template <class G>
void write_diag(FILE* f, const std::vector<pic::MacroParticle>& ps, const G& g,
                const std::vector<double>& n, double dV, double divb,
                double t) {
    double K = 0.0, W = 0.0;
    Vec3 P{0, 0, 0};
    for (const auto& mp : ps) {
        K += 0.5 * mp.p.m * mp.w * norm2(mp.p.v);
        P += mp.p.v * (mp.p.m * mp.w);
    }
    double WdB = 0.0, bz_mean = 0.0;
    for (int c = 0; c < 3; ++c) {
        double mean = 0.0;
        for (double b : g.B[c]) mean += b;
        mean /= (double)g.B[c].size();
        if (c == 2) bz_mean = mean;
        for (double b : g.B[c]) WdB += (b - mean) * (b - mean);
    }
    WdB *= dV / (2.0 * pic::MU0);
    double N = 0.0;
    for (double v : n) N += v * dV;
    for (const auto& mp : ps) W += mp.w;
    std::fprintf(f, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                 t, K, WdB, N, W, bz_mean, divb, P.x, P.y, P.z);
}

// --- self-tests (mirror pic_analysis.py P1/P2/P3/P5) ------------------------
int check(bool ok, const char* name, const std::string& detail) {
    std::printf("[%s] selftest-pic %s: %s\n", ok ? "PASS" : "FAIL", name,
                detail.c_str());
    return ok ? 0 : 1;
}

int selftest_dim(int dim) {
    int bad = 0;
    char buf[256];
    if (dim == 1) {
        pic::Grid1D g(32, 1.0e-4);
        g.set_uniform({0, 0, 0}, {0, 0, 1.0});
        // P3: uniform gather exact everywhere (incl. mid-cell Ex path).
        double worst = 0.0;
        for (int k = 0; k < 200; ++k) {
            Vec3 E, B;
            pic::gather_1d(g, (k + 0.37) * 1.0e-4 / 2, E, B);
            worst = std::max(worst, std::fabs(E.x) + std::fabs(E.y) +
                                        std::fabs(E.z) + std::fabs(B.x) +
                                        std::fabs(B.y) + std::fabs(B.z - 1.0));
        }
        std::snprintf(buf, sizeof buf, "max dev = %.3e (tol 1e-14)", worst);
        bad += check(worst < 1e-14, "1D uniform gather", buf);
        // P2: weight conservation.
        std::vector<pic::MacroParticle> ps(50);
        double wsum = 0;
        for (int k = 0; k < 50; ++k) {
            ps[(size_t)k].p.x = {k * 6.13e-5, 0, 0};
            ps[(size_t)k].p.v = {0, 0, 0};
            ps[(size_t)k].w = 1 + k;
            wsum += 1 + k;
        }
        std::vector<double> n, vx, vy, vz;
        pic::deposit_1d(ps, g, n, vx, vy, vz);
        double tot = 0;
        for (double v : n) tot += v * g.dx;
        std::snprintf(buf, sizeof buf, "deposited %.12f vs %.12f (tol 1e-9)",
                      tot, wsum);
        bad += check(std::fabs(tot - wsum) / wsum < 1e-9, "1D conservation",
                     buf);
        // P1: uniform PIC trajectory vs analytic gyro-motion (|v| conserved,
        // closes after one period). Start at the domain centre so the orbit
        // (radius r << L) never touches the periodic boundary.
        {
            pic::Grid1D gu(64, 2.0e-4);
            gu.set_uniform({0, 0, 0}, {0, 0, 1.0});
            std::vector<pic::MacroParticle> one(1);
            one[0].p.q = Q_PROTON;
            one[0].p.m = M_PROTON;
            one[0].p.x = {gu.x0 + 32 * gu.dx, 0, 0};
            one[0].p.v = {1.0e5, 0, 0};
            pic::PicOpts opt;
            opt.freeze_E = opt.freeze_B = true;
            const double om = Q_PROTON * 1.0 / M_PROTON;
            const double dt = 0.01 / om;
            const int steps = (int)std::round(2 * M_PI / (om * dt));
            std::vector<double> nn, ox, oy, oz;
            for (int s = 0; s < steps; ++s)
                pic::pic_step_1d(one, gu, dt, opt, nn, ox, oy, oz);
            double v0 = 1.0e5;
            double v1 = std::sqrt(dot(one[0].p.v, one[0].p.v));
            double r = M_PROTON * v0 / (Q_PROTON * 1.0);
            double dxc = one[0].p.x.x - (gu.x0 + 32 * gu.dx);
            std::snprintf(buf, sizeof buf,
                          "|v| err %.3e, |x(T)-x0| %.3e (r %.3e)",
                          std::fabs(v1 - v0) / v0, std::fabs(dxc), r);
            bad += check(std::fabs(v1 - v0) / v0 < 1e-9 &&
                             std::fabs(dxc) < 0.05 * r,
                         "1D uniform gyro", buf);
        }
        // P5-1D: Faraday on sine Ez gives dBy/dt = dEz/dx at mid-cells.
        {
            pic::Grid1D gs(64, 2 * M_PI / 64); // L = 2pi, k = 1
            gs.set_sine_ez(1.0, 1.0);
            gs.faraday(0.1);
            double worst2 = 0.0;
            for (int i = 0; i < gs.nx; ++i) {
                double xm = (i + 0.5) * gs.dx; // By lives at mid-cell
                double pred = 0.1 * std::cos(xm);
                worst2 = std::max(worst2, std::fabs(gs.B[1][(size_t)i] - pred));
            }
            std::snprintf(buf, sizeof buf, "max err = %.3e (tol 1e-3)",
                          worst2);
            bad += check(worst2 < 1e-3, "1D Faraday sine", buf);
        }
    } else {
        pic::Grid2D g(32, 32, 1.0e-4, 1.0e-4);
        g.set_uniform({0, 1.0e4, 0}, {0, 0, 1.0});
        double worst = 0.0;
        for (int k = 0; k < 200; ++k) {
            Vec3 E, B;
            pic::gather_2d(g, (k * 37 % 32 + 0.37) * 1.0e-4,
                           (k * 53 % 32 + 0.71) * 1.0e-4, E, B);
            worst = std::max(worst, std::fabs(E.x) + std::fabs(E.y - 1.0e4) +
                                        std::fabs(E.z) + std::fabs(B.x) +
                                        std::fabs(B.y) + std::fabs(B.z - 1.0));
        }
        std::snprintf(buf, sizeof buf, "max dev = %.3e (tol 1e-9)", worst);
        bad += check(worst < 1e-9, "2D uniform gather", buf);

        std::vector<pic::MacroParticle> ps(50);
        double wsum = 0;
        for (int k = 0; k < 50; ++k) {
            ps[(size_t)k].p.x = {k * 6.13e-5, k * 3.7e-5, 0};
            ps[(size_t)k].p.v = {0, 0, 0};
            ps[(size_t)k].w = 1 + k;
            wsum += 1 + k;
        }
        std::vector<double> n, vx, vy, vz;
        pic::deposit_2d(ps, g, n, vx, vy, vz);
        double tot = 0;
        for (double v : n) tot += v * g.dx * g.dy;
        std::snprintf(buf, sizeof buf, "deposited %.12f vs %.12f (tol 1e-9)",
                      tot, wsum);
        bad += check(std::fabs(tot - wsum) / wsum < 1e-9, "2D conservation",
                     buf);
        // P5-2D: divB stays zero through Faraday steps on random E.
        {
            pic::Grid2D gd(16, 16, 1.0e-4, 1.0e-4);
            for (int j = 0; j < 16; ++j)
                for (int i = 0; i < 16; ++i) {
                    gd.E[2][gd.id(i, j)] =
                        std::sin(i * 0.7) * std::cos(j * 1.1);
                    gd.E[0][gd.id(i, j)] = std::cos(i * 0.3 + j * 0.5);
                    gd.E[1][gd.id(i, j)] = std::sin(i * 1.3 - j * 0.2);
                }
            double d0 = gd.max_div_b(); // B=0 -> 0
            for (int s = 0; s < 10; ++s) gd.faraday(1.0e-10);
            double d1 = gd.max_div_b();
            // divB after Faraday must still be divB of the initial B (=0) up
            // to roundoff: the centred div above cancels every Ez node value
            // term by term (PDF section 5.3).
            std::snprintf(buf, sizeof buf, "divB before %.3e after %.3e",
                          d0, d1);
            bad += check(d1 < 1e-9, "2D divB conserved", buf);
        }
        // P1-2D: full gyro-orbit closes through the PIC stepper. Start at the
        // domain centre so the orbit never touches the periodic boundary.
        {
            pic::Grid2D gu(64, 64, 2.0e-4, 2.0e-4);
            gu.set_uniform({0, 0, 0}, {0, 0, 1.0});
            std::vector<pic::MacroParticle> one(1);
            one[0].p.q = Q_PROTON;
            one[0].p.m = M_PROTON;
            const double cx = gu.x0 + 32 * gu.dx, cy = gu.y0 + 32 * gu.dy;
            one[0].p.x = {cx, cy, 0};
            one[0].p.v = {1.0e5, 0, 0};
            pic::PicOpts opt;
            opt.freeze_E = opt.freeze_B = true;
            const double om = Q_PROTON * 1.0 / M_PROTON;
            const double dt = 0.01 / om;
            const int steps = (int)std::round(2 * M_PI / (om * dt));
            std::vector<double> nn, ox, oy, oz;
            for (int s = 0; s < steps; ++s)
                pic::pic_step_2d(one, gu, dt, opt, nn, ox, oy, oz);
            double clos = std::sqrt((one[0].p.x.x - cx) * (one[0].p.x.x - cx) +
                                    (one[0].p.x.y - cy) * (one[0].p.x.y - cy));
            std::snprintf(buf, sizeof buf, "|x(T)| = %.3e m (tol 5e-5)",
                          clos);
            bad += check(clos < 5e-5, "2D uniform closure", buf);
        }
    }
    return bad;
}

int selftest(int argc, char* argv[]) {
    int dim = 0;
    for (int i = 2; i < argc; ++i) {
        std::string k, v;
        if (!split_kv(argv[i], k, v)) return 1;
        if (k == "dim")
            dim = std::atoi(v.c_str());
        else {
            std::fprintf(stderr, "error: unknown key '%s'\n", k.c_str());
            return 1;
        }
    }
    int bad = 0;
    if (dim == 0 || dim == 1) bad += selftest_dim(1);
    if (dim == 0 || dim == 2) bad += selftest_dim(2);
    std::printf("selftest-pic: %s\n", bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}

int run_pic(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr, "error: --run-pic needs OUT.csv and IC.csv\n");
        return 1;
    }
    Cfg c;
    if (!parse_cfg(argc, argv, 4, c)) return 1;
    std::vector<pic::MacroParticle> ps;
    if (!read_ic(argv[3], ps, c.q0, c.m0)) return 1;

    pic::PicOpts opt;
    opt.freeze_E = c.freeze_E;
    opt.freeze_B = c.freeze_B;
    opt.kTe = c.kTe;

    FILE* out = std::fopen(argv[2], "w");
    if (!out) {
        std::fprintf(stderr, "error: cannot open '%s'\n", argv[2]);
        return 1;
    }
    std::fprintf(out, "pid,t,x,y,z,vx,vy,vz\n");
    FILE* df = nullptr;
    if (!c.diag_path.empty()) {
        df = std::fopen(c.diag_path.c_str(), "w");
        if (!df) {
            std::fprintf(stderr, "error: cannot open '%s'\n",
                         c.diag_path.c_str());
            std::fclose(out);
            return 1;
        }
        std::fprintf(df, "t,K,WdB,Ndep,Wsum,Bz_mean,max_divB,Px,Py,Pz\n");
    }
    if (c.dim == 1) {
        pic::Grid1D g(c.nx, c.dx);
        g.set_uniform(c.E0, c.B0);
        if (c.field == "sine") g.set_sine_ez(c.E0.z != 0 ? c.E0.z : 1.0e3,
                                             2 * M_PI / (c.nx * c.dx));
        if (c.field == "sinex") g.set_sine_ex(c.E0.x != 0 ? c.E0.x : 1.0e3,
                                              2 * M_PI / (c.nx * c.dx));
        std::vector<double> n, vx, vy, vz;
        write_snapshot(out, ps, 0.0);
        if (df) {
            pic::deposit_1d(ps, g, n, vx, vy, vz); // t=0 moments for row 0
            write_diag(df, ps, g, n, g.dx, 0.0, 0.0);
        }
        for (int s = 1; s <= c.steps; ++s) {
            pic::pic_step_1d(ps, g, c.dt, opt, n, vx, vy, vz);
            if (s % c.stride == 0 || s == c.steps) {
                write_snapshot(out, ps, s * c.dt);
                if (df) write_diag(df, ps, g, n, g.dx, 0.0, s * c.dt);
            }
        }
        if (!c.grid_path.empty()) {
            FILE* gf = std::fopen(c.grid_path.c_str(), "w");
            if (!gf) {
                std::fprintf(stderr, "error: cannot open '%s'\n",
                             c.grid_path.c_str());
                std::fclose(out);
                return 1;
            }
            write_grid_1d(gf, g, n, vx, vy, vz);
            std::fclose(gf);
        }
    } else {
        pic::Grid2D g(c.nx, c.ny, c.dx, c.dy);
        g.set_uniform(c.E0, c.B0);
        if (c.field == "sine")
            g.set_sine_ez(1.0e3, 2 * M_PI / (c.nx * c.dx),
                          2 * M_PI / (c.ny * c.dy));
        if (c.field == "sinex")
            g.set_sine_ex(c.E0.x != 0 ? c.E0.x : 1.0e3,
                          2 * M_PI / (c.nx * c.dx));
        std::vector<double> n, vx, vy, vz;
        write_snapshot(out, ps, 0.0);
        const double dA = g.dx * g.dy;
        if (df) {
            pic::deposit_2d(ps, g, n, vx, vy, vz); // t=0 moments for row 0
            write_diag(df, ps, g, n, dA, g.max_div_b(), 0.0);
        }
        for (int s = 1; s <= c.steps; ++s) {
            pic::pic_step_2d(ps, g, c.dt, opt, n, vx, vy, vz);
            if (s % c.stride == 0 || s == c.steps) {
                write_snapshot(out, ps, s * c.dt);
                if (df) write_diag(df, ps, g, n, dA, g.max_div_b(), s * c.dt);
            }
        }
        if (!c.grid_path.empty()) {
            FILE* gf = std::fopen(c.grid_path.c_str(), "w");
            if (!gf) {
                std::fprintf(stderr, "error: cannot open '%s'\n",
                             c.grid_path.c_str());
                std::fclose(out);
                return 1;
            }
            write_grid_2d(gf, g, n, vx, vy, vz);
            std::fclose(gf);
        }
    }
    std::fclose(out);
    if (df) std::fclose(df);
    std::fprintf(stderr, "wrote %s (pic dim=%d, n=%zu, steps=%d)\n", argv[2],
                 c.dim, ps.size(), c.steps);
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--run-pic") return run_pic(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "--selftest-pic")
        return selftest(argc, argv);
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s --run-pic OUT.csv IC.csv [dim=1|2 nx=.. ny=.. dx=.. "
                 "dy=.. dt=.. steps=.. stride=.. te=.. field=.. E=.. B=.. "
                 "freeze=.. grid=.. diag=.. q0=.. m0=..]\n"
                 "  %s --selftest-pic [dim=1|2]\n",
                 argv[0], argv[0]);
    return 1;
}
