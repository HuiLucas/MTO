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
 *     G(p) - half_t = 0     (positive sheet, outer boundary of solid wall)
 *     G(p) + half_t = 0     (negative sheet, inner boundary of solid wall)
 *
 * Both sheets are meshed in a single pass via F_wall = |G(p)| − half_t.
 * Closing at domain boundary planes (analogous to gyroid_to_stl.py's +1 padding):
 *   • Inside domain:   F_wall = |G| − half_t  (negative inside solid wall)
 *   • Just outside:    F_wall → +1             (thin taper, always positive)
 *   → Only generates closing caps where the solid wall is cut (|G| < half_t);
 *     fluid channels (|G| > half_t) remain open at the domain faces.
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
#include <CGAL/IO/OBJ.h>
#include <CGAL/Subdivision_method_3/subdivision_methods_3.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
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
    double cap_margin = 0.02; // taper width just outside domain for closing caps (mm)

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

// ── Wall implicit function ─────────────────────────────────────────────────────
//
// F_wall(p) = |G(p)| − half_t
//
//   F < 0  →  inside the solid gyroid wall  (|G| < half_t)
//   F > 0  →  in either fluid channel        (|G| > half_t)
//   F = 0  →  zero set = BOTH sheets simultaneously
//              (G = +half_t  AND  G = -half_t, two surfaces per gyroid cell)
//
// Closing at domain boundary (analogous to gyroid_to_stl.py's +1 voxel pad):
//   Inside domain (d ≥ 0):  F = |G| − half_t   (natural value)
//   Just outside  (d < 0):  F → 1  via thin linear taper over cap_margin
//
// Why this is correct:
//   • Where solid wall meets domain face (|G| < half_t → F < 0 inside):
//       F jumps to +1 just outside → zero crossing near the domain face
//       → CGAL creates a flat "closing cap" triangle ✓
//   • Where fluid channel meets domain face (|G| > half_t → F > 0 inside):
//       F is positive on both sides → no zero crossing → no cap ✓
//
// The two earlier per-sheet functors (FunPos, FunNeg) each capped the OTHER
// fluid channel's boundary too, creating a sealed box.  FunWall avoids this.

struct FunWall {
    FT operator()(Point_3 p) const {
        double x = CGAL::to_double(p.x());
        double y = CGAL::to_double(p.y());
        double z = CGAL::to_double(p.z());

        // Signed distance from domain box (positive = inside)
        double d = std::min({
            x - g.xmin, g.xmax - x,
            y - g.ymin, g.ymax - y,
            z - g.zmin, g.zmax - z
        });

        double G    = gyroid_G(x, y, z);
        double Fraw = std::abs(G) - g.half_t;   // natural wall function

        if (d >= 0.0)
            return FT(Fraw);                     // inside domain: unchanged

        if (d < -g.cap_margin)
            return FT(1.0);                      // clearly outside: fluid

        // Thin linear taper in [-cap_margin, 0): smoothly push to +1
        // so the closing caps form just at the domain face rather than
        // creating a sharp function discontinuity that could confuse the mesher.
        double t = -d / g.cap_margin;            // 0 at face, 1 at cap_margin outside
        return FT(Fraw + t * (1.0 - Fraw));      // lerp(Fraw, 1, t)
    }
};

// ── Bounding sphere ───────────────────────────────────────────────────────────
//
// The bounding sphere MUST satisfy two conditions:
//   1. It contains the entire domain (so the mesher sees all gyroid components).
//   2. Its centre has F(centre) > 0 (CGAL's requirement: centre is "outside").
//
// OLD approach: place centre one diagonal outside xmin.  Problem: the sphere
// then has radius ≈ 18 mm and the 5×2.5×10 mm domain occupies only ~0.5% of
// the sphere volume.  CGAL seeds the Delaunay refinement by random sampling
// inside the sphere, so most of the ~74 disconnected gyroid components get
// missed → large "solid" regions in the output.
//
// FIX: use a TIGHT sphere centred on a domain point that is in a FLUID channel,
// i.e., where G(p) > half_t (for F_pos).  A fluid channel point satisfies
// F_pos > 0 automatically (no fudge factor needed).  The sphere only needs to
// be large enough to reach the farthest domain corner from that centre, keeping
// it within the domain.  CGAL's random sampling now hits the domain with near
// 100% probability, guaranteeing all components are seeded.
//
// We locate a fluid channel point by scanning a coarse quarter-period grid.
// The gyroid has ~73 % fluid volume, so finding such a point takes O(1) steps.

static Sphere_3 make_bounding_sphere()
{
    // Strategy: place the sphere centre on the domain point CLOSEST to the
    // domain centroid that lies in the right fluid channel (sign*G > half_t).
    //
    // Minimising the distance from centroid → minimises the sphere radius needed
    // to reach the farthest corner → maximises the fraction of the sphere volume
    // that lies inside the domain → maximises coverage of all gyroid components.
    //
    // Example improvement:
    //   Old approach (centre 1 diag outside xmin): sphere volume ~28 500 mm³,
    //     domain fraction ~0.4 %
    //   New approach (centre near centroid):       sphere volume ~  840 mm³,
    //     domain fraction ~15 %   → ~37× more samples hit the domain
    //
    // Scan quarter-period grid; keep the candidate closest to the centroid.

    double step = M_PI / (2.0 * g.k_base);   // λ/4 ≈ 0.375 mm at default k

    double mcx = 0.5 * (g.xmin + g.xmax);    // domain centroid
    double mcy = 0.5 * (g.ymin + g.ymax);
    double mcz = 0.5 * (g.zmin + g.zmax);

    double best_cx = 1e18, best_cy = 1e18, best_cz = 1e18;
    double best_dist2 = 1e36;

    for (double x = g.xmin + step*0.5; x < g.xmax; x += step)
    for (double y = g.ymin + step*0.5; y < g.ymax; y += step)
    for (double z = g.zmin + step*0.5; z < g.zmax; z += step)
    {
        double Gval = gyroid_G(x, y, z);
        // |G| > half_t: point is in either fluid channel → FunWall > 0 ✓
        if (std::abs(Gval) > g.half_t + 0.05)
        {
            double d2 = (x-mcx)*(x-mcx) + (y-mcy)*(y-mcy) + (z-mcz)*(z-mcz);
            if (d2 < best_dist2) { best_dist2 = d2; best_cx=x; best_cy=y; best_cz=z; }
        }
    }

    double cx, cy, cz;
    if (best_dist2 < 1e35)
    {
        cx = best_cx; cy = best_cy; cz = best_cz;
        std::cout << "  Sphere centre (fluid channel, dist-to-centroid="
                  << std::sqrt(best_dist2) << " mm): ("
                  << cx << ", " << cy << ", " << cz << ")\n";
    }
    else
    {
        // Fallback: centre outside domain (always gives F=1)
        double diag = std::sqrt(
              (g.xmax-g.xmin)*(g.xmax-g.xmin)
            + (g.ymax-g.ymin)*(g.ymax-g.ymin)
            + (g.zmax-g.zmin)*(g.zmax-g.zmin));
        cx = g.xmin - diag;
        cy = mcy;
        cz = mcz;
        std::cerr << "  WARNING: no fluid-channel centre found; using fallback\n";
    }

    // Minimum radius to contain the entire domain + 5 % margin
    double rmax = 0.0;
    for (int dx : {0,1}) for (int dy : {0,1}) for (int dz : {0,1}) {
        double ex = (dx ? g.xmax : g.xmin) - cx;
        double ey = (dy ? g.ymax : g.ymin) - cy;
        double ez = (dz ? g.zmax : g.zmin) - cz;
        rmax = std::max(rmax, std::sqrt(ex*ex + ey*ey + ez*ez));
    }
    double r = rmax * 1.05;
    std::cout << "  Sphere r = " << r << " mm (volume "
              << (4.0/3.0*M_PI*r*r*r) << " mm³)\n";
    return Sphere_3(Point_3(cx, cy, cz), FT(r * r));
}

// ── Mesh the complete gyroid wall ─────────────────────────────────────────────
//
// Single call with FunWall = |G| - half_t.
// CGAL::Non_manifold_tag allows the many disconnected components (both sheets
// of each gyroid unit cell, ~74 components total) plus the flat closing caps
// that form at the domain boundary faces (only in solid-wall regions).

static SMesh mesh_wall(double angular_bound,
                       double radius_bound,
                       double distance_bound)
{
    std::cout << "Meshing gyroid wall (|G| − half_t, both sheets + closing caps) …\n"
              << std::flush;

    Sphere_3 bsphere = make_bounding_sphere();
    std::cout << "  Sphere r = "
              << std::sqrt(CGAL::to_double(bsphere.squared_radius()))
              << " mm  (volume "
              << (4.0/3.0*M_PI*std::pow(
                      std::sqrt(CGAL::to_double(bsphere.squared_radius())),3))
              << " mm³)\n";

    Tr tr;
    C2t3 c2t3(tr);

    typedef CGAL::Implicit_surface_3<GT, FunWall> Surface_3;
    Surface_3 surface(FunWall{}, bsphere);

    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(
        angular_bound, radius_bound, distance_bound);

    CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());

    SMesh sm;
    CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, sm);

    std::cout << "  Result: " << sm.num_vertices() << " vertices, "
              << sm.num_faces() << " faces\n";
    return sm;
}

// ── Shared soup-repair step ───────────────────────────────────────────────────
//
// make_surface_mesh with Non_manifold_tag can produce non-manifold edges.
// Round-trip through the polygon soup API to orient and deduplicate.
// Used by both the QuadriFlow (triangle OBJ) and Catmull-Clark (quad OBJ) paths.

static SMesh repair_mesh(const SMesh& sm_raw)
{
    namespace PMP = CGAL::Polygon_mesh_processing;

    std::vector<Point_3>                         soup_pts;
    std::vector<std::array<std::size_t, 3>>      soup_tri;
    soup_pts.reserve(sm_raw.num_vertices());
    soup_tri.reserve(sm_raw.num_faces());

    for (auto v : sm_raw.vertices())
        soup_pts.push_back(sm_raw.point(v));
    for (auto f : sm_raw.faces()) {
        auto h = sm_raw.halfedge(f);
        soup_tri.push_back({
            (std::size_t)sm_raw.target(h).idx(),
            (std::size_t)sm_raw.target(sm_raw.next(h)).idx(),
            (std::size_t)sm_raw.target(sm_raw.next(sm_raw.next(h))).idx()
        });
    }

    PMP::repair_polygon_soup(soup_pts, soup_tri);
    PMP::orient_polygon_soup(soup_pts, soup_tri);

    SMesh sm;
    PMP::polygon_soup_to_polygon_mesh(soup_pts, soup_tri, sm);

    { std::vector<SMesh::Vertex_index> iso;
      for (auto v : sm.vertices()) if (sm.is_isolated(v)) iso.push_back(v);
      for (auto v : iso) sm.remove_vertex(v); }
    PMP::remove_degenerate_faces(sm);
    sm.collect_garbage();
    return sm;
}

// ── Write repaired triangle mesh as OBJ (input for QuadriFlow) ───────────────

static void repair_and_write_obj(const SMesh& sm_raw, const std::string& out_path)
{
    std::cout << "Repairing mesh …\n" << std::flush;
    SMesh sm = repair_mesh(sm_raw);
    std::cout << "  " << sm.num_vertices() << " verts, "
              << sm.num_faces() << " faces\n";
    std::cout << "Writing triangle OBJ: " << out_path << "\n";
    if (!CGAL::IO::write_OBJ(out_path, sm))
        std::cerr << "ERROR: cannot write " << out_path << "\n";
}

// ── Write binary STL ──────────────────────────────────────────────────────────

static void write_stl(const std::string& out_path, const SMesh& sm)
{
    std::vector<Point_3>                         points;
    std::vector<std::array<std::size_t, 3>>      triangles;
    points.reserve(sm.num_vertices());
    triangles.reserve(sm.num_faces());

    for (auto v : sm.vertices())
        points.push_back(sm.point(v));
    for (auto f : sm.faces()) {
        auto h = sm.halfedge(f);
        triangles.push_back({
            (std::size_t)sm.target(h).idx(),
            (std::size_t)sm.target(sm.next(h)).idx(),
            (std::size_t)sm.target(sm.next(sm.next(h))).idx()
        });
    }
    std::cout << "Writing " << out_path << " — "
              << points.size() << " vertices, "
              << triangles.size() << " triangles\n";
    if (!CGAL::IO::write_STL(out_path, points, triangles,
                              CGAL::parameters::use_binary_mode(true)))
        std::cerr << "ERROR: cannot write " << out_path << "\n";
}

// ── Face classification + split-OBJ writer ───────────────────────────────────

enum FaceKind { SIDES = 0, PLUS_SHEET = 1, MINUS_SHEET = 2 };

static FaceKind classify_face(const SMesh& sm, SMesh::Face_index f)
{
    // Compute centroid
    auto h  = sm.halfedge(f);
    auto p0 = sm.point(sm.target(h));
    auto p1 = sm.point(sm.target(sm.next(h)));
    auto p2 = sm.point(sm.target(sm.next(sm.next(h))));
    double cx = (CGAL::to_double(p0.x()) + CGAL::to_double(p1.x()) + CGAL::to_double(p2.x())) / 3.0;
    double cy = (CGAL::to_double(p0.y()) + CGAL::to_double(p1.y()) + CGAL::to_double(p2.y())) / 3.0;
    double cz = (CGAL::to_double(p0.z()) + CGAL::to_double(p1.z()) + CGAL::to_double(p2.z())) / 3.0;

    double margin = 2.5 * g.cap_margin;
    bool near_box = (cx - g.xmin < margin || g.xmax - cx < margin ||
                     cy - g.ymin < margin || g.ymax - cy < margin ||
                     cz - g.zmin < margin || g.zmax - cz < margin);

    if (near_box) {
        // Compute face normal via cross product
        double ax = CGAL::to_double(p1.x() - p0.x());
        double ay = CGAL::to_double(p1.y() - p0.y());
        double az = CGAL::to_double(p1.z() - p0.z());
        double bx = CGAL::to_double(p2.x() - p0.x());
        double by = CGAL::to_double(p2.y() - p0.y());
        double bz = CGAL::to_double(p2.z() - p0.z());
        double nx = ay*bz - az*by;
        double ny = az*bx - ax*bz;
        double nz = ax*by - ay*bx;
        double len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12) {
            nx /= len; ny /= len; nz /= len;
            double maxcomp = std::max({std::abs(nx), std::abs(ny), std::abs(nz)});
            if (maxcomp > 0.6)
                return SIDES;
        }
    }

    return gyroid_G(cx, cy, cz) >= 0.0 ? PLUS_SHEET : MINUS_SHEET;
}

static void write_split_obj(const SMesh& sm, const std::string& base_path)
{
    // Derive stem: strip .obj if present
    std::string stem = base_path;
    if (stem.size() >= 4 && stem.substr(stem.size()-4) == ".obj")
        stem = stem.substr(0, stem.size()-4);
    // Strip trailing _tri or _all suffix if present so we get a clean stem
    for (const std::string& suf : {"_tri", "_all"}) {
        if (stem.size() > suf.size() &&
            stem.substr(stem.size()-suf.size()) == suf)
            stem = stem.substr(0, stem.size()-suf.size());
    }

    std::string paths[3] = { stem+"_sides.obj", stem+"_plus.obj", stem+"_minus.obj" };

    // Classify all faces
    std::vector<FaceKind> kind(sm.num_faces());
    int counts[3] = {0, 0, 0};
    for (auto f : sm.faces()) {
        FaceKind k = classify_face(sm, f);
        kind[f.idx()] = k;
        ++counts[(int)k];
    }
    std::cout << "  Split: SIDES=" << counts[0]
              << "  PLUS_SHEET=" << counts[1]
              << "  MINUS_SHEET=" << counts[2] << "\n";

    // Collect all vertex positions once
    std::vector<std::array<double,3>> all_verts;
    all_verts.reserve(sm.num_vertices());
    for (auto v : sm.vertices()) {
        auto p = sm.point(v);
        all_verts.push_back({CGAL::to_double(p.x()),
                             CGAL::to_double(p.y()),
                             CGAL::to_double(p.z())});
    }

    for (int k = 0; k < 3; ++k) {
        // Collect faces for this kind, build local vertex mapping
        std::vector<std::array<std::size_t,3>> faces;
        std::unordered_map<std::size_t,std::size_t> vmap; // global -> local 0-based
        std::vector<std::size_t> local_verts;

        for (auto f : sm.faces()) {
            if ((int)kind[f.idx()] != k) continue;
            auto h  = sm.halfedge(f);
            std::array<std::size_t,3> tri;
            tri[0] = sm.target(h).idx();
            tri[1] = sm.target(sm.next(h)).idx();
            tri[2] = sm.target(sm.next(sm.next(h))).idx();
            for (int j = 0; j < 3; ++j) {
                if (vmap.find(tri[j]) == vmap.end()) {
                    vmap[tri[j]] = local_verts.size();
                    local_verts.push_back(tri[j]);
                }
                tri[j] = vmap[tri[j]];
            }
            faces.push_back(tri);
        }

        std::ofstream out(paths[k]);
        if (!out) { std::cerr << "ERROR: cannot write " << paths[k] << "\n"; continue; }
        for (auto gi : local_verts) {
            out << "v " << all_verts[gi][0]
                << " "  << all_verts[gi][1]
                << " "  << all_verts[gi][2] << "\n";
        }
        for (auto& tri : faces) {
            out << "f " << (tri[0]+1) << " " << (tri[1]+1) << " " << (tri[2]+1) << "\n";
        }
        out.close();
        std::cout << "  Written: " << paths[k] << "  ("
                  << local_verts.size() << " verts, " << faces.size() << " faces)\n";
    }
}

// ── Catmull-Clark subdivision → pure quad mesh, write OBJ ────────────────────
//
// Applied DIRECTLY to the CGAL-generated curvature-adaptive triangle mesh,
// WITHOUT isotropic remeshing.  This preserves the variable triangle density:
//   - small triangles in curved gyroid regions → small quads (fine detail)
//   - large triangles in flat regions         → large quads (efficient)
//
// One CC iteration: each triangle → 3 quads, output is 100% quads.
// Isotropic remeshing (used by stl_to_quad_cgal for arbitrary STL inputs)
// destroys CGAL's curvature-adaptive density and must NOT be applied here.

static void catmullclark_and_write_obj(const SMesh& sm_raw,
                                       int   iterations,
                                       const std::string& out_path)
{
    std::cout << "Repairing mesh for Catmull-Clark …\n" << std::flush;
    SMesh sm = repair_mesh(sm_raw);
    std::cout << "  After repair: " << sm.num_vertices() << " verts, "
              << sm.num_faces() << " faces\n";

    std::cout << "Catmull-Clark subdivision (" << iterations << " iter) …\n" << std::flush;
    CGAL::Subdivision_method_3::CatmullClark_subdivision(
        sm, CGAL::parameters::number_of_iterations(iterations));

    std::size_t n_quad = 0, n_other = 0;
    for (auto f : sm.faces()) {
        std::size_t deg = 0;
        for (auto v : CGAL::vertices_around_face(sm.halfedge(f), sm)) { (void)v; ++deg; }
        (deg == 4) ? ++n_quad : ++n_other;
    }
    std::cout << "  " << sm.num_vertices() << " vertices, "
              << n_quad << " quads, " << n_other << " other\n";

    std::cout << "Writing " << out_path << " …\n";
    if (!CGAL::IO::write_OBJ(out_path, sm))
        std::cerr << "ERROR: cannot write " << out_path << "\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <params.bin> <output>"
                  << " [--angular  A]"
                  << " [--radius   R]"
                  << " [--distance D]"
                  << " [--quad]"
                  << " [--quad-iters N]"
                  << " [--split]\n"
                  << "  Without --quad: output is binary STL (triangle mesh)\n"
                  << "  With    --quad: Catmull-Clark subdivision applied;\n"
                  << "                 output is OBJ (pure quad mesh, 3×faces)\n"
                  << "  With    --split: also write _plus/_minus/_sides OBJ files\n";
        return 1;
    }

    std::string params_path = argv[1];
    std::string out_path    = argv[2];

    double angular_bound  = 30.0;
    double radius_bound   = 0.15;
    double distance_bound = 0.07;
    bool   do_quad        = false;
    int    quad_iters     = 1;
    bool   do_split       = false;

    for (int i = 3; i < argc; ++i)
    {
        std::string flag(argv[i]);
        if      (flag == "--quad")  { do_quad  = true; }
        else if (flag == "--split") { do_split = true; }
        else if (i + 1 < argc) {
            std::string val(argv[i+1]);
            if      (flag == "--angular")    { angular_bound  = std::stod(val); ++i; }
            else if (flag == "--radius")     { radius_bound   = std::stod(val); ++i; }
            else if (flag == "--distance")   { distance_bound = std::stod(val); ++i; }
            else if (flag == "--quad-iters") { quad_iters     = std::stoi(val); ++i; }
        }
    }

    g.cap_margin = std::max(0.005, radius_bound * 0.1);

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
              << " mm  cap_margin=" << g.cap_margin << " mm\n";
    if (do_quad)
        std::cout << "Quad output: Catmull-Clark " << quad_iters
                  << " iter(s) → OBJ (no isotropic remeshing)\n";

    // ── Mesh both sheets + closing caps in one pass ────────────────────────
    SMesh sm = mesh_wall(angular_bound, radius_bound, distance_bound);

    // ── Write output ───────────────────────────────────────────────────────
    // --quad:            repair + Catmull-Clark → quad OBJ
    // no flag, .obj ext: repair → triangle OBJ  (feed to QuadriFlow)
    // no flag, .stl ext: binary STL             (legacy / inspection)
    bool out_is_obj = out_path.size() >= 4 &&
                      out_path.substr(out_path.size()-4) == ".obj";

    if (do_quad)
        catmullclark_and_write_obj(sm, quad_iters, out_path);
    else if (out_is_obj) {
        repair_and_write_obj(sm, out_path);
        if (do_split) {
            std::cout << "Splitting into plus/minus/sides OBJ files …\n";
            SMesh sm_rep = repair_mesh(sm);
            write_split_obj(sm_rep, out_path);
        }
    } else {
        write_stl(out_path, sm);
    }

    std::cout << "Done.\n";
    return 0;
}
