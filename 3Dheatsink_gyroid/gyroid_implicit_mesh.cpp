/**
 * gyroid_implicit_mesh.cpp
 *
 * Mesh the thick gyroid wall surface directly from the implicit function using
 * CGAL's Surface_mesher (no voxelisation, no marching cubes).
 *
 * Physics
 * -------
 *   G(x,y,z) = sin(kx·x)·cos(ky·y) + sin(ky·y)·cos(kz·z) + sin(kz·z)·cos(kx·x)
 *
 *   kx = ky = kz = k_base + dk(p)   where dk is an RBF spatial-frequency field
 *   (with optional gyroid rotation via R)
 *
 *   Wall surfaces:
 *     G(p) - half_t = 0     (positive sheet)
 *     G(p) + half_t = 0     (negative sheet)
 *
 * The two sheets are meshed separately and then combined into one mesh file.
 *
 * CGAL meshing is intrinsically curvature-aware:
 *   • The distance_bound criterion limits how far the mesh may deviate from the
 *     true surface.  In high-curvature regions this forces more refinement.
 *   • The radius_bound limits facet circumradius (uniform density baseline).
 *
 * Binary params file (written by gyroid_to_quad_mesh.py):
 *   magic    8 B  "GYROID01"
 *   domain   8×8B  xmin xmax ymin ymax zmin zmax k_base half_t  (f64)
 *   has_rot  1 B   (0 or 1)
 *   R[9]     9×8B  rotation matrix row-major f64  (identity if has_rot=0)
 *   has_rbf  1 B   (0 or 1)
 *   If has_rbf:
 *     nx ny nz     3×4B  (int32)
 *     gx gy gz     3×8B  grid origin f64
 *     dx dy dz     3×8B  grid spacing f64
 *     data         nx×ny×nz×3  f64  (C-order, last dim = channel)
 *
 * Usage
 * -----
 *   ./gyroid_implicit_mesh <params.bin> <output.stl>
 *                          [--angular  30]   (deg, quality lower bound)
 *                          [--radius   0.15] (mm, max circumradius)
 *                          [--distance 0.07] (mm, max surface deviation)
 *
 * Build
 * -----
 *   g++ -std=c++17 -O3 gyroid_implicit_mesh.cpp -o gyroid_implicit_mesh -lgmp -lmpfr
 */

#include <CGAL/Installation/internal/disable_deprecation_warnings_and_errors.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/STL.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ── CGAL types ────────────────────────────────────────────────────────────────

typedef CGAL::Surface_mesh_default_triangulation_3          Tr;
typedef CGAL::Complex_2_in_triangulation_3<Tr>              C2t3;
typedef Tr::Geom_traits                                     GT;
typedef GT::Sphere_3                                        Sphere_3;
typedef GT::Point_3                                         Point_3;
typedef GT::FT                                              FT;
typedef CGAL::Surface_mesh<Point_3>                        SMesh;

// ── Global gyroid state ───────────────────────────────────────────────────────

struct GyrState {
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double k_base, half_t;
    double margin = 0.3;  // boundary taper width in mm (set from --margin)

    bool has_rotation;
    double R[9];   // row-major 3×3

    bool has_rbf;
    int nx, ny, nz;
    double gx_min, gy_min, gz_min;
    double gdx, gdy, gdz;
    std::vector<double> grid; // (nx × ny × nz × 3) C-order, last = channel
} g;

// ── Read binary params file ───────────────────────────────────────────────────

static bool read_params(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open params file: " << path << "\n"; return false; }

    // Magic
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "GYROID01", 8) != 0)
    { std::cerr << "Bad magic in params file\n"; return false; }

    // Domain + physics (8 doubles)
    double buf8[8];
    f.read(reinterpret_cast<char*>(buf8), 64);
    g.xmin   = buf8[0]; g.xmax   = buf8[1];
    g.ymin   = buf8[2]; g.ymax   = buf8[3];
    g.zmin   = buf8[4]; g.zmax   = buf8[5];
    g.k_base = buf8[6]; g.half_t = buf8[7];

    // Rotation
    uint8_t flag;
    f.read(reinterpret_cast<char*>(&flag), 1);
    g.has_rotation = (flag != 0);
    f.read(reinterpret_cast<char*>(g.R), 72);  // 9 doubles

    // RBF grid
    f.read(reinterpret_cast<char*>(&flag), 1);
    g.has_rbf = (flag != 0);

    if (g.has_rbf)
    {
        int32_t dims[3];
        f.read(reinterpret_cast<char*>(dims), 12);
        g.nx = dims[0]; g.ny = dims[1]; g.nz = dims[2];

        double origin[3], spacing[3];
        f.read(reinterpret_cast<char*>(origin),  24);
        f.read(reinterpret_cast<char*>(spacing), 24);
        g.gx_min = origin[0]; g.gy_min = origin[1]; g.gz_min = origin[2];
        g.gdx    = spacing[0]; g.gdy   = spacing[1]; g.gdz    = spacing[2];

        std::size_t n = (std::size_t)g.nx * g.ny * g.nz * 3;
        g.grid.resize(n);
        f.read(reinterpret_cast<char*>(g.grid.data()), n * 8);
    }

    if (!f) { std::cerr << "Truncated params file\n"; return false; }
    return true;
}

// ── RBF trilinear lookup ──────────────────────────────────────────────────────

static void rbf_lookup(double x, double y, double z, double dk[3])
{
    if (!g.has_rbf) { dk[0] = dk[1] = dk[2] = 0.0; return; }

    double gx = (x - g.gx_min) / g.gdx;
    double gy = (y - g.gy_min) / g.gdy;
    double gz = (z - g.gz_min) / g.gdz;

    int ix = std::max(0, std::min((int)gx, g.nx - 2));
    int iy = std::max(0, std::min((int)gy, g.ny - 2));
    int iz = std::max(0, std::min((int)gz, g.nz - 2));

    double tx = gx - ix, ty = gy - iy, tz = gz - iz;
    tx = std::max(0.0, std::min(1.0, tx));
    ty = std::max(0.0, std::min(1.0, ty));
    tz = std::max(0.0, std::min(1.0, tz));

    const int ny = g.ny, nz = g.nz;
    auto V = [&](int i, int j, int k, int c) -> double {
        return g.grid[((i * ny + j) * nz + k) * 3 + c];
    };

    for (int c = 0; c < 3; ++c)
        dk[c] = V(ix,  iy,  iz,  c) * (1-tx)*(1-ty)*(1-tz)
              + V(ix+1,iy,  iz,  c) *    tx *(1-ty)*(1-tz)
              + V(ix,  iy+1,iz,  c) * (1-tx)*   ty *(1-tz)
              + V(ix+1,iy+1,iz,  c) *    tx *   ty *(1-tz)
              + V(ix,  iy,  iz+1,c) * (1-tx)*(1-ty)*   tz
              + V(ix+1,iy,  iz+1,c) *    tx *(1-ty)*   tz
              + V(ix,  iy+1,iz+1,c) * (1-tx)*   ty *   tz
              + V(ix+1,iy+1,iz+1,c) *    tx *   ty *   tz;
}

// ── Gyroid G function ─────────────────────────────────────────────────────────

static double gyroid_G(double x, double y, double z)
{
    double dk[3];
    rbf_lookup(x, y, z, dk);

    double kx = g.k_base + dk[0];
    double ky = g.k_base + dk[1];
    double kz = g.k_base + dk[2];

    if (g.has_rotation)
    {
        // [u,v,w] = R @ [kx*x, ky*y, kz*z]
        double p = kx * x, q = ky * y, r = kz * z;
        double u = g.R[0]*p + g.R[1]*q + g.R[2]*r;
        double v = g.R[3]*p + g.R[4]*q + g.R[5]*r;
        double w = g.R[6]*p + g.R[7]*q + g.R[8]*r;
        return std::cos(u)*std::cos(v) + std::sin(v)*std::cos(w) - std::sin(w)*std::sin(u);
    }

    return  std::sin(kx*x)*std::cos(ky*y)
          + std::sin(ky*y)*std::cos(kz*z)
          + std::sin(kz*z)*std::cos(kx*x);
}

// ── Boundary taper ────────────────────────────────────────────────────────────
//
// Replaces the previous hard F=1 outside-domain cutoff.
//
// Problem with hard cutoff: at every domain face where G(p) < half_t, the
// function jumps from negative (solid) to +1, producing a spurious zero level
// set exactly ON the domain boundary.  CGAL meshes this "phantom surface",
// making the output look like a solid box.
//
// Fix: smooth taper over g.margin mm from the boundary.
//   d  = signed distance from domain (positive inside, negative outside)
//   s  = smoothstep(d / margin)  in [0,1]   (0 at boundary, 1 well inside)
//   F  = lerp(1, G - half_t, s)
//      = 1 + s * ((G - half_t) - 1)
//
// At d ≤ 0   (outside):  F = 1.0  (always positive, mesher ignores)
// At d = 0   (boundary): s = 0 → F = 1.0  (no zero crossing at wall)
// At d ≥ margin:         s = 1 → F = G - half_t  (natural gyroid value)
//
// The surface only exists in the interior and terminates before reaching
// the domain walls, leaving the boundary clean.

static double domain_blend_F(double x, double y, double z, double Gval, double sign)
{
    // Signed distance to the domain box  (positive = inside)
    double d = std::min({
        x - g.xmin, g.xmax - x,
        y - g.ymin, g.ymax - y,
        z - g.zmin, g.zmax - z
    });

    if (d <= 0.0) return 1.0;              // outside domain

    double m = g.margin;
    double s;
    if (d >= m)
        s = 1.0;                           // fully inside: no blending
    else {
        double t = d / m;                  // t ∈ (0,1)
        s = t * t * (3.0 - 2.0 * t);      // smoothstep: 0→1
    }

    double F_inner = sign * Gval - g.half_t;  // +1: G-half_t, -1: -G-half_t
    return 1.0 + s * (F_inner - 1.0);         // lerp(1, F_inner, s)
}

// Positive sheet:  zero set at  G(p) = +half_t
struct FunPos {
    FT operator()(Point_3 p) const {
        double x = CGAL::to_double(p.x());
        double y = CGAL::to_double(p.y());
        double z = CGAL::to_double(p.z());
        double G = gyroid_G(x, y, z);
        return FT(domain_blend_F(x, y, z, G, +1.0));
    }
};

// Negative sheet:  zero set at  G(p) = -half_t
struct FunNeg {
    FT operator()(Point_3 p) const {
        double x = CGAL::to_double(p.x());
        double y = CGAL::to_double(p.y());
        double z = CGAL::to_double(p.z());
        double G = gyroid_G(x, y, z);
        return FT(domain_blend_F(x, y, z, G, -1.0));
    }
};

// ── Bounding sphere ───────────────────────────────────────────────────────────
//
// The centre is placed one bbox-diagonal outside the xmin face, guaranteeing
// F(centre) = +1 regardless of the gyroid value at that location.
// The squared radius is chosen to encompass the entire domain with 10% margin.

static Sphere_3 make_bounding_sphere()
{
    double diag = std::sqrt(
          (g.xmax-g.xmin)*(g.xmax-g.xmin)
        + (g.ymax-g.ymin)*(g.ymax-g.ymin)
        + (g.zmax-g.zmin)*(g.zmax-g.zmin));

    double cx = g.xmin - diag;    // always outside domain
    double cy = 0.5 * (g.ymin + g.ymax);
    double cz = 0.5 * (g.zmin + g.zmax);

    // Maximum distance from centre to any domain corner
    double rmax = 0.0;
    for (int dx : {0,1}) for (int dy : {0,1}) for (int dz : {0,1}) {
        double ex = (dx ? g.xmax : g.xmin) - cx;
        double ey = (dy ? g.ymax : g.ymin) - cy;
        double ez = (dz ? g.zmax : g.zmin) - cz;
        rmax = std::max(rmax, std::sqrt(ex*ex + ey*ey + ez*ez));
    }
    double r = rmax * 1.1;        // 10% margin
    return Sphere_3(Point_3(cx, cy, cz), FT(r * r));
}

// ── Mesh one implicit surface ────────────────────────────────────────────────

template <typename Fun>
static SMesh mesh_one_sheet(const Fun& fun,
                            const Sphere_3& bsphere,
                            double angular_bound,
                            double radius_bound,
                            double distance_bound,
                            const std::string& label)
{
    std::cout << "  Meshing sheet " << label << " …\n" << std::flush;

    Tr tr;
    C2t3 c2t3(tr);

    typedef CGAL::Implicit_surface_3<GT, Fun>   Surface_3;
    Surface_3 surface(fun, bsphere);

    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(
        angular_bound,
        radius_bound,
        distance_bound
    );

    CGAL::make_surface_mesh(c2t3, surface, criteria,
                            CGAL::Non_manifold_tag());

    SMesh sm;
    CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, sm);

    std::cout << "  Sheet " << label << ": "
              << sm.num_vertices() << " vertices, "
              << sm.num_faces() << " faces\n";
    return sm;
}

// ── Combine two Surface_mesh objects and write STL ────────────────────────────

static void write_combined_stl(const std::string& out_path,
                                const SMesh& sm_pos,
                                const SMesh& sm_neg)
{
    // Collect all points and triangles from both meshes
    using P3 = Point_3;
    std::vector<P3>                          points;
    std::vector<std::array<std::size_t, 3>> triangles;
    points.reserve(sm_pos.num_vertices() + sm_neg.num_vertices());
    triangles.reserve(sm_pos.num_faces() + sm_neg.num_faces());

    // Positive sheet
    for (auto v : sm_pos.vertices())
        points.push_back(sm_pos.point(v));
    for (auto f : sm_pos.faces()) {
        auto h = sm_pos.halfedge(f);
        std::array<std::size_t, 3> tri = {
            (std::size_t)sm_pos.target(h).idx(),
            (std::size_t)sm_pos.target(sm_pos.next(h)).idx(),
            (std::size_t)sm_pos.target(sm_pos.next(sm_pos.next(h))).idx()
        };
        triangles.push_back(tri);
    }

    // Negative sheet (offset indices)
    std::size_t offset = sm_pos.num_vertices();
    for (auto v : sm_neg.vertices())
        points.push_back(sm_neg.point(v));
    for (auto f : sm_neg.faces()) {
        auto h = sm_neg.halfedge(f);
        std::array<std::size_t, 3> tri = {
            offset + (std::size_t)sm_neg.target(h).idx(),
            offset + (std::size_t)sm_neg.target(sm_neg.next(h)).idx(),
            offset + (std::size_t)sm_neg.target(sm_neg.next(sm_neg.next(h))).idx()
        };
        triangles.push_back(tri);
    }

    std::cout << "Writing combined STL: "
              << points.size() << " vertices, "
              << triangles.size() << " triangles  →  " << out_path << "\n";

    if (!CGAL::IO::write_STL(out_path, points, triangles,
                              CGAL::parameters::use_binary_mode(true)))
    {
        std::cerr << "ERROR: cannot write " << out_path << "\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <params.bin> <output.stl>"
                  << " [--angular  A]"
                  << " [--radius   R]"
                  << " [--distance D]"
                  << " [--margin   M]\n";
        return 1;
    }

    std::string params_path = argv[1];
    std::string out_path    = argv[2];

    double angular_bound  = 30.0;
    double radius_bound   = 0.15;
    double distance_bound = 0.07;
    double margin         = -1.0;   // -1 = auto (3 × radius_bound)

    for (int i = 3; i < argc - 1; ++i)
    {
        std::string flag(argv[i]), val(argv[i+1]);
        if      (flag == "--angular")  { angular_bound  = std::stod(val); ++i; }
        else if (flag == "--radius")   { radius_bound   = std::stod(val); ++i; }
        else if (flag == "--distance") { distance_bound = std::stod(val); ++i; }
        else if (flag == "--margin")   { margin         = std::stod(val); ++i; }
    }
    // Auto-set margin: 3 × radius so the taper spans roughly 3 mesh elements.
    // This is wide enough to be smooth but narrow enough not to push the surface
    // far from the domain wall.
    if (margin < 0) margin = 3.0 * radius_bound;
    g.margin = margin;

    // ── Load parameters ────────────────────────────────────────────────────
    std::cout << "Reading params: " << params_path << "\n";
    if (!read_params(params_path)) return 1;

    std::cout << "  Domain  : x[" << g.xmin << "," << g.xmax << "]"
              << "  y[" << g.ymin << "," << g.ymax << "]"
              << "  z[" << g.zmin << "," << g.zmax << "]\n"
              << "  k_base  : " << g.k_base << " rad/mm\n"
              << "  half_t  : " << g.half_t << "\n"
              << "  rotation: " << (g.has_rotation ? "yes" : "no") << "\n"
              << "  RBF grid: ";
    if (g.has_rbf)
        std::cout << g.nx << "×" << g.ny << "×" << g.nz << "\n";
    else
        std::cout << "none (uniform gyroid)\n";

    std::cout << "Meshing criteria: angular=" << angular_bound
              << "°  radius=" << radius_bound
              << " mm  distance=" << distance_bound
              << " mm  margin=" << g.margin << " mm\n";

    // ── Build bounding sphere ─────────────────────────────────────────────
    Sphere_3 bsphere = make_bounding_sphere();
    std::cout << "Bounding sphere: centre ("
              << CGAL::to_double(bsphere.center().x()) << ", "
              << CGAL::to_double(bsphere.center().y()) << ", "
              << CGAL::to_double(bsphere.center().z()) << ")  r²="
              << CGAL::to_double(bsphere.squared_radius()) << "\n";

    // ── Mesh both sheets ───────────────────────────────────────────────────
    SMesh sm_pos = mesh_one_sheet(FunPos{}, bsphere,
                                  angular_bound, radius_bound, distance_bound,
                                  "G=+half_t");
    SMesh sm_neg = mesh_one_sheet(FunNeg{}, bsphere,
                                  angular_bound, radius_bound, distance_bound,
                                  "G=-half_t");

    // ── Write combined output ──────────────────────────────────────────────
    write_combined_stl(out_path, sm_pos, sm_neg);

    std::cout << "Done.\n";
    return 0;
}
