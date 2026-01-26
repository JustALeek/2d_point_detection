import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, LineString, Point
from shapely.ops import split, linemerge, unary_union
from scipy.interpolate import splprep, splev
import shapely
import os
import mariadb

# ============================================================
# DATA LOADING & PARSING
# ============================================================

class DataLoader:
    @staticmethod
    def load_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return img

    @staticmethod
    def parse_xml(xml_path, image_path, point_xml_path=None):
        """
        Parse CVAT-style XML annotation for the image whose name matches image_path.

        Returns:
            polygons      : list of dict {id, polygon, label}
            points        : stitching points
            inner_points  : inner stitching points
        """
        image_name = os.path.basename(image_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_tag = None
        for img_tag in root.findall("image"):
            if img_tag.attrib.get("name") == image_name:
                image_tag = img_tag
                break

        if image_tag is None:
            raise ValueError(f"No matching <image> entry found in XML for {image_name}")

        polygons = []
        for i, poly in enumerate(image_tag.findall(".//polygon")):
            coords = [tuple(map(float, p.split(",")))
                      for p in poly.attrib["points"].split(";")]
            label = poly.attrib.get("label", "unknown")
            polygons.append({"id": i, "polygon": Polygon(coords), "label": label})

        if point_xml_path:
            tree = ET.parse(point_xml_path)
            root = tree.getroot()

            image_tag = None
            for img_tag in root.findall("image"):
                if img_tag.attrib.get("name") == image_name:
                    image_tag = img_tag
                    break
        
        points, inner_points, overlap_points = [], [], []
        for pt in image_tag.findall(".//points"):
            label = pt.attrib["label"]
            point_pairs = [p for p in pt.attrib["points"].split(";") if p.strip()]

            if label == "point":
                container = points
            elif label == "inner":
                container = inner_points
            else:
                container = overlap_points

            for pair in point_pairs:
                x, y = map(float, pair.split(","))
                container.append({
                    "label": label,
                    "point": Point(x, y)
                })
        resolved_overlap, debug_fits = DataLoader.resolve_overlaps(overlap_points, points, inner_points, polygons)
        points.extend([p for p in resolved_overlap if p["label"] == "point"])
        inner_points.extend([p for p in resolved_overlap if p["label"] == "inner"])

        # slider_values = {}
        # for slider_value in root.findall("slider_variables"):
        #     slider_values[slider_value.attrib["name"]] = slider_value.attrib["value"]

        return polygons, points, inner_points, debug_fits

    @staticmethod
    def get_2d_width(polygons):
        all_bounds = [polygon["polygon"].bounds for polygon in polygons]
        global_min_x = min(b[0] for b in all_bounds)
        global_max_x = max(b[2] for b in all_bounds)
        return global_max_x - global_min_x
    
    @staticmethod
    def best_model_distance(point, neighbors):
        """
        Returns the smallest distance between a point and the neighbours' PCA line & quadratic curve
        """
        best = np.inf

        # PCA line
        pca = StitchingProcessor.fit_line_pca(neighbors)
        if pca:
            d = StitchingProcessor.point_line_distance(point, pca[0], pca[1])
            best = min(best, d)

        # Quadratic
        quad = StitchingProcessor.fit_local_quadratic(neighbors)
        if quad:
            d = quad(point)
            best = min(best, d)

        return best
    
    def resolve_overlaps(overlap_points,points,inner_points, polygons, k=6, confidence_ratio=0.4):
        """
        Iteratively resolve overlap points.

        Each resolved overlap point becomes part of the neighbor set.
        """

        # Convert to simple Point lists
        point_neighbors = [p["point"] for p in points]
        inner_neighbors = [p["point"] for p in inner_points]

        # Sort overlap points by proximity to existing points
        def nearest_labeled_distance(op):
            p = op["point"]
            d1 = min([p.distance(q) for q in point_neighbors], default=np.inf)
            d2 = min([p.distance(q) for q in inner_neighbors], default=np.inf)
            return min(d1, d2)

        overlap_sorted = sorted(overlap_points, key=nearest_labeled_distance)

        resolved = []
        debug_geometries = []

        for op in overlap_sorted:
            p = op["point"]

            # Recollect knn each time an overlay point is processed
            pn = StitchingProcessor.k_nearest_neighbors(p, point_neighbors, k, polygons)
            inn = StitchingProcessor.k_nearest_neighbors(p, inner_neighbors, k, polygons)

            d_p, geom_p = GeometryProcessor.best_model_with_geometry(p, pn)
            d_i, geom_i = GeometryProcessor.best_model_with_geometry(p, inn)

            if d_p < confidence_ratio * d_i:
                op["label"] = "point"
                point_neighbors.append(p)
                debug_geometries.append(geom_p)
            else:
                op["label"] = "inner"
                inner_neighbors.append(p)
                debug_geometries.append(geom_i)

            resolved.append(op)

        return resolved, debug_geometries
    
    #@staticmethod
    #def save_visualization_to_database():
        # conn = mariadb.connect(
        #     user="db_user",
        #     password="db_user_passwd",
        #     host="somehost",
        #     database="something")
        # cur = conn.cursor() 
        #!!!DO NOT RUN TS UNTIL FURTHER CONFIRMATION!!!
    
# ============================================================
# GEOMETRY PROCESSING
# ============================================================
    
class GeometryProcessor:
    """
    Geometric utilities: polygon containment, simplification, splitting.
    """
    @staticmethod
    def pca_line_geometry(points, length=200):
        res = StitchingProcessor.fit_line_pca(points)
        if not res:
            return None

        centroid, direction = res
        p1 = centroid - direction * length
        p2 = centroid + direction * length
        return LineString([tuple(p1), tuple(p2)])
    
    @staticmethod
    def quadratic_geometry(points, samples=50):
        if len(points) < 3:
            return None

        coords = np.array([[p.x, p.y] for p in points])
        min_v, max_v = coords.min(axis=0), coords.max(axis=0)

        horizontal = (max_v[0] - min_v[0]) > (max_v[1] - min_v[1])

        try:
            if horizontal:
                poly = np.poly1d(np.polyfit(coords[:, 0], coords[:, 1], 2))
                xs = np.linspace(min_v[0], max_v[0], samples)
                ys = poly(xs)
            else:
                poly = np.poly1d(np.polyfit(coords[:, 1], coords[:, 0], 2))
                ys = np.linspace(min_v[1], max_v[1], samples)
                xs = poly(ys)

            return LineString(np.column_stack([xs, ys]))

        except np.linalg.LinAlgError:
            return None
    
    @staticmethod
    def best_model_with_geometry(point, neighbors):
        """
        Returns:
            best_distance, best_geometry
        """
        best_d = np.inf
        best_geom = None

        # PCA line
        pca_geom = GeometryProcessor.pca_line_geometry(neighbors)
        if pca_geom:
            d = pca_geom.distance(point)
            if d < best_d:
                best_d = d
                best_geom = pca_geom

        # Quadratic
        quad_geom = GeometryProcessor.quadratic_geometry(neighbors)
        if quad_geom:
            d = quad_geom.distance(point)
            if d < best_d:
                best_d = d
                best_geom = quad_geom

        return best_d, best_geom

    @staticmethod
    def assign_points_to_polygons(points, polygons):
        mapping = {}
        for idx, pt in enumerate(points):
            assigned_poly = None
            for j, poly in enumerate(polygons):
                if poly["polygon"].covers(pt["point"]):
                    assigned_poly = j
                    break

            if assigned_poly is not None:
                mapping.setdefault(assigned_poly, []).append(idx)

        return mapping

    
    @staticmethod
    def combined_geometry(polygons, buffer_distance):
        """
        Create cleaned Layer2 geometry:
        - union of layer2/layer3
        - subtract background

        Returns:
            MultiPolygon for layer2 components
        """
        layer2 = unary_union([p["polygon"] 
                              for p in polygons if p["label"] in ("layer2", "layer3")]
                              )
        background = unary_union([p["polygon"] 
                                  for p in polygons if p["label"] in ("layer0", "background")]
                                  )

        return layer2.buffer(buffer_distance).difference(background)
    
    @staticmethod
    def calculate_triangle_area(p1, p2, p3):
        """
        Triangle area for polygon simplification.
        """
        return 0.5 * abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        )

    @staticmethod
    def simplify_to_quad(coords):
        """
        Reduce polygon boundary to 4 dominant vertices (quad).
        Used to approximate rectangular components.
        """
        pts = list(coords)
        while len(pts) > 4:
            min_area = float("inf")
            remove_idx = -1
            for i in range(len(pts)):
                area = GeometryProcessor.calculate_triangle_area(
                    pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
                )
                if area < min_area:
                    min_area = area
                    remove_idx = i
            del pts[remove_idx]
        return np.array(pts)

    @staticmethod
    def split_polygon_into_lines(polygon, quad_points=None):
        """
        Split polygon boundary into line segments using quad vertices.
        """
        coords = np.asarray(polygon.exterior.coords)[:-1]
        if quad_points is None:
            quad_points = GeometryProcessor.simplify_to_quad(coords)

        boundary = polygon.boundary
        cutters = MultiPoint(quad_points)
        segments = list(split(boundary, cutters).geoms)

        # Merge first/last if split wraps around
        if len(segments) > len(cutters.geoms):
            first, last = segments.pop(0), segments.pop(-1)
            segments.append(linemerge([last, first]))

        return segments
    
    @staticmethod
    def center_distance(f1, f2):
        """
        Distance between centers of two fitted outlines.
        """
        c1 = np.array([np.mean(f1['x']), np.mean(f1['y'])])
        c2 = np.array([np.mean(f2['x']), np.mean(f2['y'])])
        return np.linalg.norm(c1 - c2)
    
# ============================================================
# STITCHING POINT PROCESSING
# ============================================================

class StitchingProcessor:
    """
    Ordering and grouping stitching points along boundaries.
    """
    @staticmethod
    def k_nearest_neighbors(center_pt, candidates, k, polygons):
        """
        Returns the k nearest candidate points to center_pt in a set radius
        """
        max_radius = DataLoader.get_2d_width(polygons)/20
        candidates = [p for p in candidates if p.distance(center_pt) <= max_radius]
        candidates = sorted(candidates, key=lambda p: p.distance(center_pt))
        return candidates[:k]
            
    @staticmethod
    def fit_line_pca(points):
        """
        Fit a line using PCA.
        points: list of shapely Points
        Returns: (centroid, direction_unit_vector)
        """
        if len(points) < 2:
            return None

        coords = np.array([[p.x, p.y] for p in points])
        centroid = coords.mean(axis=0)

        # PCA via SVD
        _, _, Vt = np.linalg.svd(coords - centroid)
        direction = Vt[0]          # principal direction
        direction /= np.linalg.norm(direction)

        return centroid, direction
    
    @staticmethod
    def fit_local_quadratic(points):
        """
        Fit a local quadratic curve.
        Returns a callable distance function.
        """
        if len(points) < 3:
            return None

        coords = np.array([[p.x, p.y] for p in points])
        min_v, max_v = coords.min(axis=0), coords.max(axis=0)

        # Decide dominant axis
        horizontal = (max_v[0] - min_v[0]) > (max_v[1] - min_v[1])

        try:
            if horizontal:
                coeffs = np.polyfit(coords[:, 0], coords[:, 1], 2)
                poly = np.poly1d(coeffs)

                def distance_fn(pt):
                    x, y = pt.x, pt.y
                    return abs(y - poly(x))

            else:
                coeffs = np.polyfit(coords[:, 1], coords[:, 0], 2)
                poly = np.poly1d(coeffs)

                def distance_fn(pt):
                    x, y = pt.x, pt.y
                    return abs(x - poly(y))

            return distance_fn

        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def point_line_distance(point, centroid, direction):
        """
        Perpendicular distance from point to PCA line
        """
        p = np.array([point.x, point.y])
        v = p - centroid
        proj = np.dot(v, direction) * direction
        perp = v - proj
        return np.linalg.norm(perp)
    
    def process_points_by_outline(points_xy, ring):
        """
        Reorders a scattered list of points to follow the sequence of the polygon boundary.
        
        Logic:
        1. Project every point onto the boundary line (finding distance from start & projected point on the boundary).
        2. Sort points based on that linear distance.
        """
        mapping = []
        for point in points_xy:
            d = ring.project(point)

            mapping.append({
                "point": point,
                "sorting_distance": d,
                "projected_point": ring.interpolate(d),
                "distance": point.distance(ring)
            })

        mapping_sorted = sorted(mapping, key = lambda x: x["sorting_distance"])
        
        return mapping_sorted

    @staticmethod
    def process_point_groups(polygons, points, mapping, combined_geom):
        """
        Group points by outline and generate ordered connections.
        """
        combined_components = (list(combined_geom.geoms) if isinstance(combined_geom, MultiPolygon) else [combined_geom])

        map_point_to_projection_outline = {}

        for poly_idx, pt_indices in mapping.items():
            polygon = polygons[poly_idx]
            label = polygon["label"]
            pts = [points[i]["point"] for i in pt_indices]

            # Case A: Standard layers - use the polygon's own boundary
            if polygon["label"] != "layer2":
                map_point_to_projection_outline[(polygon["polygon"].boundary, label)] = pts

            # Case B: Layer 2 - use the constructed combined geometry components
            else:
                # Find WHICH part of the MultiPolygon the point belongs to
                for layer2_component in combined_components:
                    # Check the first point (assumption: group belongs to same component)
                    if layer2_component.covers(points[pt_indices[0]]["point"]):
                        map_point_to_projection_outline.setdefault((layer2_component.boundary, label), []).extend(pts)

        connected = {}

        for ring_info, pts in map_point_to_projection_outline.items():
            # Order the points so they form a continuous line along the shape
            ring = ring_info[0]
            processed_pts = StitchingProcessor.process_points_by_outline(pts, ring)

            n = len(processed_pts)
            
            # Skip if there are too few points to form a connection
            if n < 2:
                continue
            
            # Close the loop: connect the last point back to the first
            processed_pts.append(processed_pts[0]) 
            connected[ring_info] = processed_pts

        return connected
    
    @staticmethod
    def compute_distance_along_connections(processed_pts):
        """
        Compute distances between consecutive stitching points.
        """
        dists = []
        for i in range(len(processed_pts)-1):
            dists.append(processed_pts[i+1]["point"].distance(processed_pts[i]["point"]))

        med_dist = np.median(dists)

        return dists, med_dist

# ============================================================
# COMPONENT ALIGNMENT (OVERLAY ↔ MUDGUARD)
# ============================================================

class ComponentProcessor:
    """
    Fit component outlines and measure inter-component misalignment.
    """
    @staticmethod
    def fit_outline(segment, trim_ratio=0.25, extension_ratio=0.3):
        """
        Fit quadratic curve to a boundary segment and extend it.
        """
        coords = np.array(segment.coords)
        if len(coords) < 2:
            return None

        min_v, max_v = np.min(coords, axis=0), np.max(coords, axis=0)
        orientation = 'horizontal' if (max_v[0] - min_v[0]) > (max_v[1] - min_v[1]) else 'vertical'

        dense = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            steps = max(2, int(np.linalg.norm(p2 - p1)))
            for k in range(steps):
                dense.append(p1 + (k / steps) * (p2 - p1))
        dense.append(coords[-1])
        arr = np.array(dense)

        trim = int(len(arr) * trim_ratio)
        if len(arr) > 2 * trim + 5:
            arr = arr[trim:-trim]

        try:
            if orientation == 'vertical':
                coeffs = np.polyfit(arr[:, 1], arr[:, 0], 2)
            else:
                coeffs = np.polyfit(arr[:, 0], arr[:, 1], 2)
            poly = np.poly1d(coeffs)
        except:
            return None

        if orientation == 'vertical':
            iv = coords[:, 1]
        else:
            iv = coords[:, 0]

        iv_min, iv_max = iv.min(), iv.max()
        sample = np.linspace(iv_min, iv_max, 100)
        pred = poly(sample)

        d_iv = np.diff(sample)
        d_dv = np.diff(pred)
        arc_length = np.sum(np.sqrt(d_iv**2 + d_dv**2))

        extension_length = arc_length * extension_ratio

        deriv_func = poly.deriv()
        slope_min = deriv_func(iv_min)
        slope_max = deriv_func(iv_max)

        delta_min = extension_length / np.sqrt(1 + slope_min**2)
        delta_max = extension_length / np.sqrt(1 + slope_max**2)

        extended_iv_start = iv_min - delta_min
        extended_iv_end = iv_max + delta_max

        extrapolated_iv = np.linspace(extended_iv_start, extended_iv_end, 100)
        extrapolated_dv = poly(extrapolated_iv)

        if orientation == 'vertical':
            x, y = extrapolated_dv, extrapolated_iv
        else:
            x, y = extrapolated_iv, extrapolated_dv

        return {
            'x': x,
            'y': y,
            'shapely': LineString(np.column_stack((x, y)))
        }
    
    @staticmethod
    def quad_projected_distance(quad_points, overlay_fit, mudguard_fit):
        """
        quad_points: np.ndarray (4,2)  # mudguard quad
        ov_fit, mud_fit: fitted line dicts
        """
        overlay_line = overlay_fit['shapely']
        mudguard_line = mudguard_fit['shapely']

        dists = []

        for q in quad_points:
            qp = Point(q)

            # project quad point to overlay line
            overlay_proj_dist = overlay_line.project(qp)
            overlay_proj_pt = overlay_line.interpolate(overlay_proj_dist)

            # project quad point to mudguard line
            mudguard_proj_dist = mudguard_line.project(qp)
            mudguard_proj_pt = mudguard_line.interpolate(mudguard_proj_dist)

            # distance between projected points
            dists.append(overlay_proj_pt.distance(mudguard_proj_pt))

        # representative distance (min = closest structural alignment)
        return min(dists)
    
    @staticmethod
    def alignment_match(polygons):
        """
        Match overlay outlines to mudguard outlines.

        Returns:
            matches : list of best-fit overlay↔mudguard line pairs
            stitching_alignment_closest_boundary : overlay boundaries closest to mudguard
        """
        mudguards = [p for p in polygons if p["label"]=="layer0"]
        overlays = [p for p in polygons if p["label"]=="layer3"]

        matches, stitching_alignment_closest_boundary = [], []
        
        mid, old = 1, 1   # unique line IDs

        for mudguard in mudguards:
            # Approximate mudguard shape with quad
            coords = np.asarray(mudguard["polygon"].exterior.coords)[:-1]
            quad = GeometryProcessor.simplify_to_quad(coords)

            # Fit mudguard boundary segments
            mudguard_fits = []
            for line in GeometryProcessor.split_polygon_into_lines(mudguard["polygon"], quad):
                fitted_line = ComponentProcessor.fit_outline(line)
                if fitted_line:
                    fitted_line["id"] = mid
                    mid += 1
                    mudguard_fits.append(fitted_line)

            # Match closest overlay components
            for overlay in sorted(overlays, key=lambda x: mudguard["polygon"].distance(x["polygon"]))[:2]:
                overlay_fits = []
                for line in GeometryProcessor.split_polygon_into_lines(overlay["polygon"]):
                    fitted_line = ComponentProcessor.fit_outline(line)
                    if fitted_line:
                        fitted_line["id"] = old
                        old += 1
                        overlay_fits.append(fitted_line)

                # Generate all candidate pairings
                pairs = []
                for overlay_fit in overlay_fits:
                    for mudguard_fit in mudguard_fits:
                        pairs.append({
                            "overlay": overlay_fit,
                            "mudguard": mudguard_fit,
                            "center": GeometryProcessor.center_distance(overlay_fit, mudguard_fit),
                            "distance": ComponentProcessor.quad_projected_distance(quad, overlay_fit, mudguard_fit)
                        })

                if not pairs:
                    continue
                
                # Greedy matching with exclusivity
                start = len(matches)
                used_overlay, used_mudguard = set(), set()

                pairs.sort(key=lambda x: x["center"])
                p = pairs[0]
                used_overlay.add(p["overlay"]["id"])
                used_mudguard.add(p["mudguard"]["id"])
                matches.append({
                    "overlay_line": p["overlay"],
                    "mudguard_line": p["mudguard"],
                    "distance": p["distance"]
                })

                stitching_alignment_closest_boundary.append(p["overlay"]["shapely"])

                # Additional valid matches
                for r in sorted(pairs, key=lambda x:x["distance"]):
                    if r["overlay"]["id"] in used_overlay or r["mudguard"]["id"] in used_mudguard:
                        continue
                    used_overlay.add(r["overlay"]["id"])
                    used_mudguard.add(r["mudguard"]["id"])
                    matches.append({
                        "overlay_line": r["overlay"],
                        "mudguard_line": r["mudguard"],
                        "distance": r["distance"]
                    })

                # Remove worst extra match (enforce 1:1)
                if len(matches) - start > 1:
                    idx = max(range(start, len(matches)), key=lambda x: matches[x]["distance"])
                    matches.pop(idx)

        return matches, stitching_alignment_closest_boundary
    
class StitchingAlignmentProcessor:
    """
    Detects misalignment between stitching lines and component boundaries.
    """
    @staticmethod
    def alignment_check(connected_points, stitching_alignment_closest_boundary, stitching_alignment_candidates):
        """
        Identify stitching segments that should be checked for alignment error.
        """
        # Extract overlay quad points
        quad_points = []
        for ring_info, processed_pts in connected_points.items():
            if ring_info[1] == "layer3":
                coords = [processed_pt["point"].coords[0] for processed_pt in processed_pts]
                quad_points.extend(GeometryProcessor.simplify_to_quad(coords))

        # Select closest stitching candidate to each overlay boundary
        stitching_lines_to_check = []
        for boundary in stitching_alignment_closest_boundary:
            distances = []
            for line in stitching_alignment_candidates:
                distances.append((line, shapely.distance(boundary.centroid, line.centroid)))
            distances = sorted(distances, key = lambda x:x[1])
            stitching_lines_to_check.append(distances[0][0])

        # Measure distances from stitching lines to quad points
        stitching_alignment_to_check = []
        for line in stitching_lines_to_check:
            distances = []
            for point in quad_points:
                distances.append(line.distance(Point(point)))
            distances = sorted(distances, key = lambda x:x)
            stitching_alignment_to_check.append([line, distances[:2]])

        return stitching_alignment_to_check
    
class VisualizationProcessor:
    """
    Visualization utilities for stitching and component alignment errors.
    """
    @staticmethod
    def visualize_best_fit_lines(vis, geometries, color=(0, 255, 0), thickness=1):
        """
        Visualizes best fit lines used to sort overlap points.
        """
        for geom in geometries:
            if geom is None:
                continue

            coords = np.array(geom.coords, dtype=int)
            for i in range(len(coords) - 1):
                cv2.line(
                    vis,
                    tuple(coords[i]),
                    tuple(coords[i + 1]),
                    color,
                    thickness
                )
        return vis
    
    @staticmethod
    def visualize_stitching_error(vis, connected_points, neighbour_margin, boundary_margin, max_line_dist, type='points'):
        """
        Visualize stitching point spacing and boundary deviation.
        """
        stitching_alignment_candidates = []

        for ring_info, processed_pts in connected_points.items():
            
            med_boundary_dist = np.median([x["distance"] for x in processed_pts])
            neighbour_dist, med_neighbour_dist = StitchingProcessor.compute_distance_along_connections(processed_pts)

            num_connected_lines = [0] * len(processed_pts)

            # Draw connections
            for i in range(len(processed_pts)-1):
                cur_pt = processed_pts[i]
                p1, p2 = cur_pt["point"], processed_pts[i+1]["point"]

                if neighbour_dist[i] < max_line_dist:
                    num_connected_lines[i] += 1
                    num_connected_lines[i+1] += 1

                    spi_ok = abs(neighbour_dist[i] - med_neighbour_dist) < neighbour_margin
                    color, size = ((255, 0, 0), 1) if spi_ok else ((0, 0, 255), 1)
                    cv2.line(vis, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color, size)
                
                # Candidate stitching misalignment
                elif type=='points' and ring_info[1] == "layer2" and neighbour_dist[i] < max_line_dist * 5.3:
                    stitching_alignment_candidates.append(LineString([p1, p2]))

            # Draw points & projections
            for i, cur_pt in enumerate(processed_pts):
                p1 = cur_pt["point"]
                margin_ok = (abs(cur_pt["distance"] - med_boundary_dist) < boundary_margin) or num_connected_lines[i] < 2
                color, size = ((255, 0, 0), 1) if margin_ok else ((0, 0, 255), 1)
                cv2.circle(vis, (int(p1.x), int(p1.y)), size, color, -1)
                
                projected_pt = cur_pt["projected_point"]
                if p1.distance(projected_pt) < max_line_dist:
                    cv2.line(vis, (int(p1.x), int(p1.y)), (int(projected_pt.x), int(projected_pt.y)), (0, 255, 255), 1)
                    cv2.circle(vis, (int(projected_pt.x), int(projected_pt.y)), 1, (0, 0, 0), 1) 

        return vis, stitching_alignment_candidates
    
    @staticmethod
    def visualize_component_alignment_error(vis, matches, max_component_offset_distance):
        """
        Visualize overlay ↔ mudguard alignment result.
        """
        for match in matches:
            alignment_ok = match["distance"] < max_component_offset_distance
            color, size = ((255, 0, 0), 1) if alignment_ok else ((0, 0, 255), 3)

            overlay_points = np.column_stack((match["overlay_line"]["x"], match["overlay_line"]["y"])).astype(int)
            for i in range(len(overlay_points) - 1):
                cv2.line(vis, tuple(overlay_points[i]), tuple(overlay_points[i + 1]), color, size)

            mudguard_points = np.column_stack((match["mudguard_line"]["x"], match["mudguard_line"]["y"])).astype(int)
            for i in range(len(mudguard_points) - 1):
                cv2.line(vis, tuple(mudguard_points[i]), tuple(mudguard_points[i + 1]), (0, 0, 0), 3)
        
        return vis
    
    @staticmethod
    def visualize_stitching_alignment_error(vis, stitching_alignment_to_check, max_stitching_offset_distance):
        """
        Highlight stitching segments exceeding allowed offset.
        """
        for stitching_dist in stitching_alignment_to_check:
            if stitching_dist[1][0] > max_stitching_offset_distance or stitching_dist[1][1] > max_stitching_offset_distance:
                stitching_pts = np.vstack(stitching_dist[0].coords[:]).astype(int)
                cv2.line(vis, tuple(stitching_pts[0]), tuple(stitching_pts[1]), (0, 0, 255), 3)
        
        return vis
    
    @staticmethod
    def draw_visualization(vis_ax,
           img,
           points,
           inner_points,
           matches,
           stitching_alignment_closest_boundary,
           neighbour_margin_factor,
           boundary_margin_factor,
           max_connected_line_dist,
           max_component_offset_distance,
           max_stitching_offset_distance):
        # Visualization base
        alpha = 0.7
        vis = img.copy()
        vis = cv2.addWeighted(vis, alpha, np.full_like(img, 255), 1 - alpha, 0)

        # Draw best fit lines
        #vis = VisualizationProcessor.draw_best_fit_lines(vis, debug_fits)

        # Draw stitching
        vis, stitching_alignment_candidates = VisualizationProcessor.visualize_stitching_error(vis, points, neighbour_margin_factor, boundary_margin_factor, max_connected_line_dist, 'points')
        vis, _ = VisualizationProcessor.visualize_stitching_error(vis, inner_points, neighbour_margin_factor, boundary_margin_factor, max_connected_line_dist, 'inner points')

        # Draw component alignment
        vis = VisualizationProcessor.visualize_component_alignment_error(vis, matches, max_component_offset_distance)

        # Stitching alignment
        stitching_alignment_to_check = StitchingAlignmentProcessor.alignment_check(points, stitching_alignment_closest_boundary, stitching_alignment_candidates)
        vis = VisualizationProcessor.visualize_stitching_alignment_error(vis, stitching_alignment_to_check, max_stitching_offset_distance)
        
        # Final output
        vis_ax.clear()
        vis_ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        vis_ax.axis("off")
        
# ============================================================
# VISUALIZATION PIPELINE
# ============================================================

def visualize_2d_error(image_path, xml_path, point_xml_path=None, width_2d=None, save_path=None):
    img = DataLoader.load_image(image_path)

    polygons, points, inner_points, debug_fits = DataLoader.parse_xml(xml_path, image_path, point_xml_path)

    if width_2d is None:
        width_2d = DataLoader.get_2d_width(polygons)

    buffer_distance = int(width_2d / 402)
    neighbour_margin_factor = 330
    boundary_margin_factor = 330
    max_connected_line_dist = 40
    max_component_offset_distance = 210
    max_stitching_offset_distance = 630
    
    mapping_points = GeometryProcessor.assign_points_to_polygons(points, polygons)
    mapping_inner_points = GeometryProcessor.assign_points_to_polygons(inner_points, polygons)
    combined = GeometryProcessor.combined_geometry(polygons, buffer_distance)
    
    connected_points = StitchingProcessor.process_point_groups(polygons, points, mapping_points, combined)
    connected_inner_points = StitchingProcessor.process_point_groups(polygons, inner_points, mapping_inner_points, combined)
    #TODO: These should also be able to be loaded directly from db
    
    # Component alignment
    matches, stitching_alignment_closest_boundary = ComponentProcessor.alignment_match(polygons)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.05, right =0.95, top=0.95, bottom=0.25)

    ax_neigh = plt.axes([0.2, 0.21, 0.6, 0.03])
    ax_bound = plt.axes([0.2, 0.17, 0.6, 0.03])
    ax_line  = plt.axes([0.2, 0.13, 0.6, 0.03])
    ax_comp  = plt.axes([0.2, 0.09, 0.6, 0.03])
    ax_stitch= plt.axes([0.2, 0.05, 0.6, 0.03])

    s_neigh = Slider(ax_neigh, "Neighbour Margin", 165, 660, valinit=neighbour_margin_factor)
    s_bound = Slider(ax_bound, "Boundary Margin", 165, 660, valinit=boundary_margin_factor)
    s_line  = Slider(ax_line,  "Max Line Dist", 20, 80, valinit=max_connected_line_dist)
    s_comp  = Slider(ax_comp, "Max Comp Offset Dist", 105, 420, valinit=max_component_offset_distance)
    s_stitch= Slider(ax_stitch, "Max Stitching Offset Dist", 315, 1260, valinit=max_stitching_offset_distance)

    def update(val):
        s_neigh.valtext.set_text(f"width_2d({int(width_2d)})/{int(s_neigh.val)}")
        s_bound.valtext.set_text(f"width_2d({int(width_2d)})/{int(s_bound.val)}")
        s_line.valtext.set_text(f"width_2d({int(width_2d)})/{int(s_line.val)}")
        s_stitch.valtext.set_text(f"width_2d({int(width_2d)})/{int(s_stitch.val)}")

        VisualizationProcessor.draw_visualization(ax,
                                                  img,
                                                  connected_points,
                                                  connected_inner_points,
                                                  matches,
                                                  stitching_alignment_closest_boundary,
                                                  int(width_2d/s_neigh.val),
                                                  int(width_2d/s_bound.val),
                                                  int(width_2d/s_line.val),
                                                  int(width_2d/s_comp.val),
                                                  int(width_2d/s_stitch.val))
        fig.canvas.draw_idle()
    
    s_neigh.on_changed(update)
    s_bound.on_changed(update)
    s_line.on_changed(update)
    s_comp.on_changed(update)
    s_stitch.on_changed(update)
    update(0.0)

    def save_data(val):
        print("SAVE BUTTON WAS CLICKED")
        # DataLoader.save_visualization_to_xml()

    ax_bsave = plt.axes([0.925, 0.1, 0.05, 0.1])
    bsave = Button(ax_bsave, 'Save', color="grey")
    bsave.on_clicked(save_data)

    def open_image(val):
        print("OPEN BUTTON WAS CLICKED")

    ax_bopen = plt.axes([0.925, 0.2, 0.05, 0.1])
    bopen = Button(ax_bopen, 'Open', color="grey")
    bopen.on_clicked(open_image)

    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    image_path = r"C:\Users\user\Downloads\2d_point_detection\orig_img\IMG_161122.jpg"
    xml_path = r"C:\Users\user\Downloads\2d_point_detection\annotations.xml"
    point_xml_path = r"C:\Users\user\Downloads\2d_point_detection\pred\IMG_161122.xml"  # optional

    visualize_2d_error(
        image_path,
        xml_path,
        point_xml_path=point_xml_path
    )