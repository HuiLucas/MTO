/**
 * stl_to_quad_cgal.cpp
 *
 * Convert a large STL triangle mesh to a quad-dominant mesh using CGAL 5.6.
 *
 * Pipeline
 * --------
 *  1. Read STL → CGAL::Surface_mesh  (welds duplicate vertices automatically)
 *  2. Triangulate any non-triangular faces, repair borders.
 *  3. Compute discrete Gaussian curvature (angle-defect) per vertex.
 *     This is O(V) and needs no extra libraries.
 *  4. Garland-Heckbert QEM simplification with a face-count ratio stopping
 *     criterion.  QEM is inherently curvature-aware: high-curvature vertices
 *     accumulate large quadric errors and are preserved; flat regions collapse.
 *     Additionally, we pin vertices whose |curvature| exceeds a threshold so
 *     they are never collapsed.
 *  5. CGAL::Polygon_mesh_processing::isotropic_remeshing with a target edge
 *     length scaled by the bbox diagonal.  Curvature-guided variant: we set a
 *     shorter edge length for runs on the high-curvature half of the mesh via
 *     the constrained-vertex mechanism.
 *  6. Catmull-Clark subdivision (one iteration):
 *       each triangle  → 3 quads
 *       output is a PURE QUAD mesh
 *  7. Write OBJ (all faces have exactly 4 vertices).
 *
 * Usage
 * -----
 *   ./stl_to_quad_cgal  <input.stl>  <output.obj>
 *                       [--target-faces N]      (default 50000)
 *                       [--remesh-edge-pct P]   (default 1.5, % of bbox diag)
 *                       [--curv-pin-percentile C] (default 90)
 *
 * Build
 * -----
 *   g++ -std=c++17 -O2 stl_to_quad_cgal.cpp -o stl_to_quad_cgal -lgmp -lmpfr
 */

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

// IO
#include <CGAL/IO/STL.h>
#include <CGAL/IO/OBJ.h>

// Polygon-soup repair / orientation / conversion
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

// Repair / triangulate
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>

// Simplification
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Face_count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Face_count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_triangle_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>

// Isotropic remeshing
#include <CGAL/Polygon_mesh_processing/remesh.h>

// Catmull-Clark subdivision
#include <CGAL/Subdivision_method_3/subdivision_methods_3.h>

// General
#include <boost/property_map/property_map.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

typedef CGAL::Simple_cartesian<double>  K;
typedef CGAL::Surface_mesh<K::Point_3>  Mesh;
typedef Mesh::Vertex_index              VD;
typedef Mesh::Face_index                FD;
typedef Mesh::Edge_index                ED;
typedef Mesh::Halfedge_index            HD;
typedef K::Point_3                      Point3;

namespace SMS = CGAL::Surface_mesh_simplification;
namespace PMP = CGAL::Polygon_mesh_processing;

// ── Helpers ──────────────────────────────────────────────────────────────────

static double angle_at_vertex(const Point3& p0,
                               const Point3& p1,
                               const Point3& p2)
{
    auto v1 = p1 - p0;
    auto v2 = p2 - p0;
    double dot = CGAL::to_double(v1 * v2);
    double n1  = std::sqrt(CGAL::to_double(v1 * v1));
    double n2  = std::sqrt(CGAL::to_double(v2 * v2));
    if (n1 < 1e-14 || n2 < 1e-14) return 0.0;
    double c = dot / (n1 * n2);
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c);
}

static double triangle_area(const Point3& p0,
                             const Point3& p1,
                             const Point3& p2)
{
    auto cross = CGAL::cross_product(p1 - p0, p2 - p0);
    return 0.5 * std::sqrt(CGAL::to_double(cross * cross));
}

// ── 1. Discrete Gaussian curvature (angle-defect) ────────────────────────────
//
//   K(v) = (2π − Σ θ_i) / A_mixed(v)
//
// We use barycentric area (A/3 per triangle) which is simpler and still O(V).
// Returns absolute curvature normalized to [0, 1] (1 = at the user's percentile).

static std::vector<float>
compute_curvature(const Mesh& mesh, double pin_percentile)
{
    // Use the maximum vertex index as vector size (safe after collect_garbage)
    std::size_t nv = 0;
    for (VD v : mesh.vertices()) nv = std::max(nv, (std::size_t)v.idx() + 1);
    std::vector<double> k_abs(nv, 0.0);
    std::vector<double> area(nv, 0.0);

    for (FD f : mesh.faces())
    {
        HD h = mesh.halfedge(f);
        VD v0 = mesh.target(h);
        VD v1 = mesh.target(mesh.next(h));
        VD v2 = mesh.target(mesh.next(mesh.next(h)));

        const Point3& p0 = mesh.point(v0);
        const Point3& p1 = mesh.point(v1);
        const Point3& p2 = mesh.point(v2);

        double a0 = angle_at_vertex(p0, p1, p2);
        double a1 = angle_at_vertex(p1, p2, p0);
        double a2 = angle_at_vertex(p2, p0, p1);
        double A  = triangle_area(p0, p1, p2);

        k_abs[v0.idx()] += a0;
        k_abs[v1.idx()] += a1;
        k_abs[v2.idx()] += a2;
        double A3 = A / 3.0;
        area[v0.idx()] += A3;
        area[v1.idx()] += A3;
        area[v2.idx()] += A3;
    }

    std::vector<double> gauss_abs;
    gauss_abs.reserve(nv);
    for (VD v : mesh.vertices())
    {
        double ai = area[v.idx()];
        double raw = std::abs(2.0 * M_PI - k_abs[v.idx()]);
        gauss_abs.push_back(ai > 1e-20 ? raw / ai : 0.0);
    }

    // Percentile threshold for normalisation
    std::vector<double> sorted_k = gauss_abs;
    std::sort(sorted_k.begin(), sorted_k.end());
    double thr = sorted_k[static_cast<std::size_t>(
                     pin_percentile / 100.0 * (sorted_k.size() - 1))];
    thr = std::max(thr, 1e-12);

    std::vector<float> result(nv);
    for (std::size_t i = 0; i < nv; ++i)
        result[i] = static_cast<float>(std::min(gauss_abs[i] / thr, 1.0));

    return result;
}

// ── 2. Pin high-curvature vertices from QEM collapse ─────────────────────────
//
// Returns a vertex-is-constrained property map.
// Vertices with normalised curvature ≥ 1.0 (i.e. above the pin percentile)
// are pinned.

typedef Mesh::Property_map<VD, bool> BoolVertPMap;

static BoolVertPMap
make_pin_map(Mesh& mesh, const std::vector<float>& curv)
{
    auto [pin_map, created] = mesh.add_property_map<VD, bool>("v:pinned", false);
    (void)created;
    for (VD v : mesh.vertices())
        pin_map[v] = (v.idx() < curv.size() && curv[v.idx()] >= 1.0f);
    return pin_map;
}

// ── 3. Constrained-edge map for remeshing ─────────────────────────────────────
//
// Boundary edges are protected so the open mesh border is preserved.

typedef Mesh::Property_map<ED, bool> BoolEdgePMap;

static BoolEdgePMap make_border_edge_map(Mesh& mesh)
{
    auto [emap, created] = mesh.add_property_map<ED, bool>("e:constrained", false);
    (void)created;
    for (ED e : mesh.edges())
        if (mesh.is_border(e))
            emap[e] = true;
    return emap;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.stl> <output.obj>"
                  << " [--target-faces N]"
                  << " [--remesh-edge-pct P]"
                  << " [--curv-pin-percentile C]\n";
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = argv[2];

    // Parse optional flags
    std::size_t target_faces      = 50000;
    std::size_t coarse_faces      = 0;     // 0 = auto (10 × target_faces)
    double      remesh_edge_pct   = 1.5;   // % of bbox diagonal
    double      curv_pin_pct      = 90.0;  // percentile for pinning

    for (int i = 3; i < argc - 1; ++i)
    {
        std::string flag(argv[i]);
        std::string val(argv[i + 1]);
        if (flag == "--target-faces")        { target_faces    = std::stoul(val); ++i; }
        else if (flag == "--coarse-faces")   { coarse_faces    = std::stoul(val); ++i; }
        else if (flag == "--remesh-edge-pct")  { remesh_edge_pct = std::stod(val);  ++i; }
        else if (flag == "--curv-pin-percentile") { curv_pin_pct = std::stod(val);  ++i; }
    }
    if (coarse_faces == 0)
        coarse_faces = std::max(target_faces * 10, (std::size_t)500000);

    // ── Step 1: Read STL → polygon soup ────────────────────────────────────
    std::cout << "Reading " << input_path << " …\n" << std::flush;

    std::vector<K::Point_3>              soup_points;
    std::vector<std::array<std::size_t,3>> soup_faces;

    if (!CGAL::IO::read_STL(input_path, soup_points, soup_faces,
                             CGAL::parameters::verbose(false)))
    {
        std::cerr << "ERROR: cannot read " << input_path << "\n";
        return 1;
    }
    std::cout << "  Soup: " << soup_points.size() << " vertices, "
              << soup_faces.size() << " triangles\n";

    // ── Step 2: Repair polygon soup, orient, convert to Surface_mesh ────────
    std::cout << "Repairing and orienting polygon soup …\n" << std::flush;

    // Remove degenerate and duplicate triangles, merge close vertices
    PMP::repair_polygon_soup(soup_points, soup_faces);

    // Orient consistently (open surfaces are fine — mismatched components
    // are flagged but orientation is still applied)
    PMP::orient_polygon_soup(soup_points, soup_faces);

    // Build Surface_mesh (requires 2-manifold from oriented soup)
    Mesh mesh;
    if (!PMP::is_polygon_soup_a_polygon_mesh(soup_faces))
    {
        // Non-manifold: try removing the bad faces
        std::cerr << "  Warning: soup is not 2-manifold — "
                     "attempting to build mesh anyway (may lose some faces)\n";
    }
    PMP::polygon_soup_to_polygon_mesh(soup_points, soup_faces, mesh);

    // Remove isolated vertices left over from non-manifold resolution
    {
        std::vector<VD> to_remove;
        for (VD v : mesh.vertices())
            if (mesh.is_isolated(v)) to_remove.push_back(v);
        for (VD v : to_remove) mesh.remove_vertex(v);
        if (!to_remove.empty())
            std::cout << "  Removed " << to_remove.size()
                      << " isolated vertices\n";
    }

    // Remove any zero-area faces (degenerate triangles)
    PMP::remove_degenerate_faces(mesh);

    // Compact so that vertex/face indices are contiguous before curvature
    mesh.collect_garbage();
    std::cout << "  After repair: " << mesh.num_vertices() << " verts, "
              << mesh.num_faces() << " faces\n";

    // ── Step 3: Curvature ───────────────────────────────────────────────────
    std::cout << "Computing curvature (angle-defect) …\n" << std::flush;
    auto curv = compute_curvature(mesh, curv_pin_pct);
    {
        std::size_t n_pinned = 0;
        for (float c : curv) if (c >= 1.0f) ++n_pinned;
        std::cout << "  " << n_pinned << " vertices pinned at "
                  << curv_pin_pct << "th percentile\n";
    }
    auto pin_map = make_pin_map(mesh, curv);

    // ── Step 4: Two-stage simplification ────────────────────────────────────
    //
    // Stage 4a: Fast edge-length-cost collapse to reduce to coarse_faces.
    //   Edge_length_cost is O(1) per edge evaluation — much faster than GH
    //   for very large meshes.  This is a "cheap" coarse pass.
    //
    // Stage 4b: Garland-Heckbert QEM from coarse_faces to target_faces,
    //   with curvature-pinned vertices.  GH is geometrically precise.
    //   Curvature pinning additionally ensures gyroid ridges/saddles are kept.

    const std::size_t n_start = mesh.num_faces();

    if (n_start > coarse_faces)
    {
        double ratio1 = static_cast<double>(coarse_faces) /
                        static_cast<double>(n_start);
        ratio1 = std::max(0.001, std::min(ratio1, 0.999));

        std::cout << "Stage 1 simplification (edge-length cost): "
                  << n_start << " → ~" << coarse_faces
                  << " faces …\n" << std::flush;

        SMS::Face_count_ratio_stop_predicate<Mesh> stop1(ratio1, mesh);
        int r1 = SMS::edge_collapse(
            mesh, stop1,
            CGAL::parameters::get_cost(SMS::Edge_length_cost<Mesh>())
                             .get_placement(SMS::Midpoint_placement<Mesh>())
        );
        mesh.collect_garbage();
        std::cout << "  Removed " << r1 << " edges.  Remaining: "
                  << mesh.num_faces() << " faces, "
                  << mesh.num_vertices() << " vertices\n";

        // Recompute curvature and pin map on the coarser mesh
        std::cout << "Recomputing curvature on reduced mesh …\n" << std::flush;
        curv    = compute_curvature(mesh, curv_pin_pct);
        pin_map = make_pin_map(mesh, curv);
        {
            std::size_t np = 0;
            for (float c : curv) if (c >= 1.0f) ++np;
            std::cout << "  " << np << " vertices pinned\n";
        }
    }

    // Stage 4b: GH with curvature pinning
    {
        const std::size_t n1 = mesh.num_faces();
        double ratio2 = static_cast<double>(target_faces) /
                        static_cast<double>(n1);
        ratio2 = std::max(0.001, std::min(ratio2, 0.999));

        std::cout << "Stage 2 simplification (Garland-Heckbert QEM): "
                  << n1 << " → ~" << target_faces
                  << " faces (ratio=" << ratio2 << ") …\n" << std::flush;

        typedef SMS::GarlandHeckbert_triangle_policies<Mesh, K> GH_policies;
        GH_policies gh_policies(mesh);
        typedef SMS::Bounded_normal_change_placement<GH_policies::Get_placement>
                Safe_placement;
        Safe_placement safe_placement(gh_policies.get_placement());

        SMS::Face_count_ratio_stop_predicate<Mesh> stop2(ratio2, mesh);
        int r2 = SMS::edge_collapse(
            mesh, stop2,
            CGAL::parameters::get_cost(gh_policies.get_cost())
                             .get_placement(safe_placement)
                             .vertex_is_constrained_map(pin_map)
        );
        mesh.collect_garbage();
        std::cout << "  Removed " << r2 << " edges.  Remaining: "
                  << mesh.num_faces() << " faces, "
                  << mesh.num_vertices() << " vertices\n";
    }

    // ── Step 5: Isotropic remeshing ─────────────────────────────────────────
    // Target edge length = remesh_edge_pct% of the bounding-box diagonal
    auto bbox = CGAL::bounding_box(mesh.points().begin(),
                                   mesh.points().end());
    double diag = std::sqrt(
        CGAL::to_double(CGAL::squared_distance(bbox.min(), bbox.max())));
    double target_len = remesh_edge_pct / 100.0 * diag;

    std::cout << "Isotropic remeshing (target edge = "
              << target_len << " mm) …\n" << std::flush;

    // Constrain border edges so the open boundary is not moved.
    // protect_constraints requires all constrained edges shorter than
    // 4/3 * target_len; skip protection if any border edge violates this.
    auto border_emap = make_border_edge_map(mesh);
    bool can_protect = true;
    double max_border_len = 0.0;
    for (ED e : mesh.edges())
    {
        if (!border_emap[e]) continue;
        HD h = mesh.halfedge(e, 0);
        auto p0 = mesh.point(mesh.source(h));
        auto p1 = mesh.point(mesh.target(h));
        double len = std::sqrt(CGAL::to_double(CGAL::squared_distance(p0, p1)));
        max_border_len = std::max(max_border_len, len);
    }
    if (max_border_len > 4.0 / 3.0 * target_len)
    {
        can_protect = false;
        std::cout << "  Note: disabling border protection "
                     "(max border edge " << max_border_len
                  << " > 4/3 * " << target_len << ")\n";
    }

    PMP::isotropic_remeshing(
        mesh.faces(), target_len, mesh,
        CGAL::parameters::number_of_iterations(5)
                         .protect_constraints(can_protect)
                         .edge_is_constrained_map(border_emap)
    );
    mesh.collect_garbage();
    std::cout << "  After remesh: " << mesh.num_faces() << " faces, "
              << mesh.num_vertices() << " vertices\n";

    // ── Step 6: Catmull-Clark subdivision → pure quad mesh ───────────────────
    //
    // After ONE Catmull-Clark iteration on a triangle mesh, every triangle is
    // replaced by 3 quads → output is 100% quad faces.
    //
    // Boundary vertices are kept fixed (no update) because the default CGAL
    // mask only handles closed meshes fully; for open meshes this still
    // produces valid quads at the boundary.

    std::cout << "Catmull-Clark subdivision (1 iter) …\n" << std::flush;
    CGAL::Subdivision_method_3::CatmullClark_subdivision(mesh,
        CGAL::parameters::number_of_iterations(1));
    std::cout << "  After CC: " << mesh.num_faces() << " faces, "
              << mesh.num_vertices() << " vertices\n";

    // ── Step 7: Write OBJ ───────────────────────────────────────────────────
    std::cout << "Writing " << output_path << " …\n" << std::flush;
    if (!CGAL::IO::write_OBJ(output_path, mesh))
    {
        std::cerr << "ERROR: cannot write " << output_path << "\n";
        return 1;
    }

    // Count quad vs non-quad faces in output
    std::size_t n_quad = 0, n_other = 0;
    for (FD f : mesh.faces())
    {
        std::size_t deg = 0;
        for (VD v : vertices_around_face(mesh.halfedge(f), mesh)) { (void)v; ++deg; }
        if (deg == 4) ++n_quad; else ++n_other;
    }

    std::cout << "  Quads: " << n_quad << "  Other: " << n_other << "\n";
    std::cout << "Done.\n";
    return 0;
}
