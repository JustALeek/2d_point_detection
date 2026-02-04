import json
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, LineString, Point, MultiLineString
from shapely.ops import split, linemerge, unary_union
from shapely.wkt import loads
import shapely
import mariadb

# Handles saving, loading, and running pipelines
class MainManager:
    def __init__(self):
        self.conn = mariadb.connect(
            user="testuser",
            password="testpass",
            host="127.0.0.1",
            database="testdb")
    
    def save_dbdata(self, img_name, img_data, polygons, connected_points, connected_inner_points, slider_values, matches, sacb):
        cur = self.conn.cursor()

        #Saving images
        sql = """
        INSERT INTO images (img_name, img_data)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            img_data = VALUES(img_data)
        """
        cur.execute(sql, (img_name, img_data))

        #Saving polygons
        sql = """
        INSERT INTO polygons (img_name, polygon_index, label, vertices)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            label = VALUES(label),
            vertices = VALUES(vertices)
        """
        
        for poly_dict in polygons:
            polygon_index = poly_dict["id"]
            label = poly_dict.get("label", "")
            polygon_obj = poly_dict["polygon"]
            
            # Convert Shapely Polygon to list of coordinates
            vertices_list = list(polygon_obj.exterior.coords)
            
            # Convert to JSON string for storage
            vertices_json = json.dumps(vertices_list)
            
            cur.execute(sql, (img_name, polygon_index, label, vertices_json))

        # Saving connected_points and connected_inner_points
        self.save_point_data("connected_points", connected_points, img_name)
        self.save_point_data("connected_inner_points", connected_inner_points, img_name)

        # Saving slider_values
        sql = """
        INSERT INTO slider_values (
            img_name,
            neighbour_margin_factor,
            boundary_margin_factor,
            max_connected_line_dist,
            max_component_offset_distance,
            max_stitching_offset_distance
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            neighbour_margin_factor = VALUES(neighbour_margin_factor),
            boundary_margin_factor = VALUES(boundary_margin_factor),
            max_connected_line_dist = VALUES(max_connected_line_dist),
            max_component_offset_distance = VALUES(max_component_offset_distance),
            max_stitching_offset_distance = VALUES(max_stitching_offset_distance)
        """

        cur.execute(
            sql,
            (
                img_name,
                slider_values.get("neighbour_margin_factor"),
                slider_values.get("boundary_margin_factor"),
                slider_values.get("max_connected_line_dist"),
                slider_values.get("max_component_offset_distance"),
                slider_values.get("max_stitching_offset_distance")
            )
        )

        # Saving matches
        serializable_matches = []
        for m in matches:
            entry = {}
            for key in ["overlay_line", "mudguard_line"]:
                line = m.get(key)
                if line is None:
                    entry[key] = None
                    continue
                # Convert numpy arrays to lists
                entry[key] = {
                    "x": line["x"].tolist() if hasattr(line["x"], "tolist") else list(line["x"]),
                    "y": line["y"].tolist() if hasattr(line["y"], "tolist") else list(line["y"]),
                    "id": line.get("id"),
                    "shapely": line["shapely"].wkt if isinstance(line["shapely"], LineString) else line["shapely"]
                }
            entry["distance"] = float(m.get("distance", 0))
            serializable_matches.append(entry)

        match_json = None if not serializable_matches else json.dumps(serializable_matches)

        sql = """
            INSERT INTO matches (img_name, match_data)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE
                match_data = VALUES(match_data)
        """
        cur.execute(sql, (img_name, match_json))

        # Saving stitching_alignment_closest_boundary
        lines = []

        for line in sacb:
            if isinstance(line, LineString):
                lines.append(line.wkt)
            elif isinstance(line, str):
                lines.append(line)
            else:
                raise TypeError("Invalid line type")

        sql = """
        INSERT INTO sacb_results (img_name, lines_json)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE lines_json = VALUES(lines_json)
        """

        if not lines:
            cur.execute(sql, (img_name, None))
        else:
            cur.execute(sql, (img_name, json.dumps(lines)))

        self.conn.commit()

    def save_point_data(self, table_name, points, img_name):
        cur = self.conn.cursor()
        sql = f"""
        INSERT INTO {table_name} (
            img_name, boundary_linestring, layer,
            point_data
        )
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            boundary_linestring = VALUES(boundary_linestring),
            layer = VALUES(layer),
            point_data = VALUES(point_data)
        """
        
        for (linestring_obj, layer), points_list in points.items():
            if isinstance(linestring_obj, str):
                linestring_obj = loads(linestring_obj)
            # Convert LineString to list of coordinates and JSON-encode
            if isinstance(linestring_obj, LineString):
                linestring_coords = list(linestring_obj.coords)
            elif isinstance(linestring_obj, MultiLineString):
                # Flatten all coordinates from all LineStrings into a single list
                linestring_coords = []
                for ls in linestring_obj.geoms:
                    linestring_coords.extend(ls.coords)
            else:
                raise TypeError("Expected LineString or MultiLineString")
            linestring_json = json.dumps(linestring_coords)
                        
            point_data_json = json.dumps([
                {
                    "point": [pt_dict["point"].x, pt_dict["point"].y],
                    "sorting_distance": float(pt_dict["sorting_distance"]),
                    "projected_point": [pt_dict["projected_point"].x, pt_dict["projected_point"].y]
                                    if pt_dict.get("projected_point") else None,
                    "distance": float(pt_dict["distance"]) if pt_dict.get("distance") is not None else None
                }
                for pt_dict in points_list
            ])

            cur.execute(sql, (
                img_name,
                linestring_json,
                layer,
                point_data_json
            ))

    def load_dbdata(self, img_name):
        cur = self.conn.cursor()
        #Loading image
        cur.execute("SELECT img_data FROM images WHERE img_name = %s", (img_name,))
        row = cur.fetchone()
        img_bytes = row[0]
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #Loading polygons
        cur.execute("SELECT polygon_index, label, vertices FROM polygons WHERE img_name = %s", (img_name,))
        rows = cur.fetchall()
        polygons = [
            {"id": r[0], "label": r[1], "polygon": Polygon(json.loads(r[2]))} 
            for r in rows
        ]

        #Loading points
        connected_points = self.load_point_data(img_name, "connected_points")
        connected_inner_points = self.load_point_data(img_name, "connected_inner_points")

        #Loading slider values
        cur2 = self.conn.cursor(dictionary = True)
        cur2.execute("""
            SELECT neighbour_margin_factor,
                boundary_margin_factor,
                max_connected_line_dist,
                max_component_offset_distance,
                max_stitching_offset_distance
            FROM slider_values
            WHERE img_name = %s
        """, (img_name,))

        row = cur2.fetchone()

        slider_values = {key: float(value) if value is not None else None for key, value in row.items()}

        #Loading matches
        cur.execute("SELECT match_data FROM matches WHERE img_name = %s", (img_name,))
        row = cur.fetchone()

        if not row or row[0] is None:
            matches = []
        else:
            match_data = row[0]
            if isinstance(match_data, str):
                matches = json.loads(match_data)
            else:
                matches = match_data

        cur.execute("SELECT line_data FROM stitching_alignment_closest_boundary WHERE img_name = %s", (img_name,))

        row = cur.fetchone()

        # Default to empty list
        sacb = []

        if row and row[0] is not None:
            line_data = row[0]

            # If stored as JSON string
            if isinstance(line_data, str):
                wkt_list = json.loads(line_data)
            else:
                wkt_list = line_data  # already decoded JSON

            # Reconstruct Shapely LineStrings
            sacb = [loads(wkt) for wkt in wkt_list]
        return img_data, polygons, connected_points, connected_inner_points, slider_values, matches, sacb
    
    def load_point_data(self, img_name, table_name):
        cur = self.conn.cursor()
        cur.execute(f"""
            SELECT boundary_linestring, layer, point_data
            FROM {table_name}
            WHERE img_name = %s
        """, (img_name,))

        points = {}
        for boundary_json, layer, point_data_json in cur:
            # Convert JSON string to list of coordinates
            if isinstance(boundary_json, str):
                coords = json.loads(boundary_json)
            else:
                coords = boundary_json  # already a list

            line = LineString(coords)
            if line.is_empty:
                continue
            key = (line.wkt, layer)

            # Parse point_data JSON and reconstruct points
            point_list = []
            point_data_list = json.loads(point_data_json)
            for pt_dict in point_data_list:
                pt = Point(pt_dict["point"])
                proj = Point(pt_dict["projected_point"]) if pt_dict.get("projected_point") else None
                point_list.append({
                    "point": pt,
                    "sorting_distance": float(pt_dict["sorting_distance"]),
                    "projected_point": proj,
                    "distance": float(pt_dict["distance"]) if pt_dict.get("distance") is not None else None
                })
            points[key] = point_list
        return points
    
    def run_save_pipeline(self, img_name, img_data, points, inner_points, overlap_points, polygons):
        """
        Given inferred data from point_detection and segmentation, postprocess and save data to database.
        
        :param img_name: name of the formatted image
        :param img_data: image data as NumPy ndarray
        :param points: points with label "point"
        :param inner_points: points with label "inner"
        :param overlap_points: points with label "overlap"
        :param polygons: polygonal data - lineStrings matched with labels 
        """
        points = DataProcessor.to_shapely_points(points)
        inner_points = DataProcessor.to_shapely_points(inner_points)
        overlap_points = DataProcessor.to_shapely_points(overlap_points)
        polygons = GeometryProcessor.reformat_to_polygon_objects(polygons)
        polygons, connected_points, connected_inner_points, slider_values, matches, sacb = DataProcessor.process_raw_points(points, inner_points, overlap_points, polygons)
        try:
            self.save_dbdata(img_name, img_data, polygons, connected_points, connected_inner_points, slider_values, matches, sacb)
            return True
        except Exception as e:
            return False

    
    def run_load_pipeline(self, img_name):
        img, polygons, points, inner_points, slider_values, matches, sacb = self.load_dbdata(img_name)
        nmf = slider_values["neighbour_margin_factor"]
        bmf = slider_values["boundary_margin_factor"]
        mcld = slider_values["max_connected_line_dist"]
        mcod = slider_values["max_component_offset_distance"]
        msod = slider_values["max_stitching_offset_distance"]
        # Visualization base
        alpha = 0.7
        vis = img.copy()
        vis = cv2.addWeighted(vis, alpha, np.full_like(img, 255), 1 - alpha, 0)

        # Draw stitching
        vis, stitching_alignment_candidates = (
            VisualizationProcessor.visualize_stitching_error(vis, points, nmf, bmf, mcld, 'points')
            if points else (vis, [])
        )
        vis, _ = (
            VisualizationProcessor.visualize_stitching_error(vis, inner_points, nmf, bmf, mcld, 'inner_points')
            if inner_points else (vis, [])
        )

        # Draw component alignment
        vis = VisualizationProcessor.visualize_component_alignment_error(vis, matches, mcod) if matches else vis

        # Stitching alignment
        stitching_alignment_to_check = StitchingProcessor.alignment_check(points, sacb, stitching_alignment_candidates) if sacb and stitching_alignment_candidates else []
        vis = VisualizationProcessor.visualize_stitching_alignment_error(vis, stitching_alignment_to_check, msod) if stitching_alignment_to_check else vis

        return vis
        
class DataProcessor:
    """
    Handles processes related to parsing, modifying, restructuring data
    """
    def process_raw_points(points, inner_points, overlap_points, polygons):
        points, inner_points, debug_fits = StitchingProcessor.resolve_overlaps(points, inner_points, overlap_points, polygons)
        width_2d = GeometryProcessor.get_2d_width(polygons)
        buffer_distance = 402
        #initial slider values
        slider_values = {
            "neighbour_margin_factor": 330,
            "boundary_margin_factor": 330,
            "max_connected_line_dist": 40,
            "max_component_offset_distance": 210,
            "max_stitching_offset_distance": 630
        }

        mapping_points = GeometryProcessor.assign_points_to_polygons(points, polygons)
        mapping_inner_points = GeometryProcessor.assign_points_to_polygons(inner_points, polygons)
        combined = GeometryProcessor.combined_geometry(polygons, width_2d/buffer_distance)
        
        connected_points = StitchingProcessor.process_point_groups(polygons, points, mapping_points, combined)
        connected_inner_points = StitchingProcessor.process_point_groups(polygons, inner_points, mapping_inner_points, combined)
        matches, stitching_alignment_closest_boundary = ComponentProcessor.alignment_match(polygons)
        return polygons, connected_points, connected_inner_points, slider_values, matches, stitching_alignment_closest_boundary

    @staticmethod
    def to_shapely_points(points):
        shapely_points = []
        for p in points:
            shapely_points.append(Point(p[0], p[1]))
        return shapely_points
    
# ============================================================
# GEOMETRY PROCESSING
# ============================================================

class GeometryProcessor:
    """
    Geometric utilities: polygon containment, simplification, splitting.
    """    
    @staticmethod
    def get_2d_width(polygons):
        all_bounds = [polygon["polygon"].bounds for polygon in polygons]
        global_min_x = min(b[0] for b in all_bounds)
        global_max_x = max(b[2] for b in all_bounds)
        return global_max_x - global_min_x
    
    @staticmethod
    def reformat_to_polygon_objects(raw_polygons):
        label_mapping={
            0: 'background',
            1: 'layer0',
            2: 'layer1',
            3: 'layer2',
            4: 'layer3',
            5: 'swoosh'
            }
        contours = raw_polygons.get("contours", [])
        labels = raw_polygons.get("labels", [])

        if len(contours) != len(labels):
            raise ValueError("Contours and labels length mismatch")

        objects = []

        for idx, (contour, label) in enumerate(zip(contours, labels)):
            # Shapely requires at least 3 unique points
            if contour is None or len(contour) < 3:
                continue

            poly = Polygon(contour)

            # Skip empty or invalid results
            if poly.is_empty:
                continue

            objects.append({
                "id": idx,
                "polygon": poly,
                "label": label_mapping[label]
            })

        return objects
    
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
                if poly["polygon"].covers(pt):
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
    def resolve_overlaps(points, inner_points, overlap_points, polygons, k=6, confidence_ratio=0.4):
        """
        Iteratively resolve overlap points.

        Each resolved overlap point becomes part of the neighbor set.
        """

        # Sort overlap points by proximity to existing points
        def nearest_labeled_distance(p):
            d1 = min([p.distance(q) for q in points], default=np.inf)
            d2 = min([p.distance(q) for q in inner_points], default=np.inf)
            return min(d1, d2)

        overlap_sorted = sorted(overlap_points, key=nearest_labeled_distance)
        debug_geometries = []

        for p in overlap_sorted:
            # Recollect knn each time an overlay point is processed
            pn = StitchingProcessor.k_nearest_neighbors(p, points, k, polygons)
            inn = StitchingProcessor.k_nearest_neighbors(p, inner_points, k, polygons)

            d_p, geom_p = GeometryProcessor.best_model_with_geometry(p, pn)
            d_i, geom_i = GeometryProcessor.best_model_with_geometry(p, inn)

            if d_p < confidence_ratio * d_i:
                points.append(p)
                debug_geometries.append(geom_p)
            else:
                inner_points.append(p)
                debug_geometries.append(geom_i)
        return points, inner_points, debug_geometries
    
    @staticmethod
    def k_nearest_neighbors(center_pt, candidates, k, polygons):
        """
        Returns the k nearest candidate points to center_pt in a set radius
        """
        max_radius = GeometryProcessor.get_2d_width(polygons)/20
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
            pts = [points[i] for i in pt_indices]

            # Case A: Standard layers - use the polygon's own boundary
            if polygon["label"] != "layer2":
                map_point_to_projection_outline[(polygon["polygon"].boundary, label)] = pts

            # Case B: Layer 2 - use the constructed combined geometry components
            else:
                # Find WHICH part of the MultiPolygon the point belongs to
                for layer2_component in combined_components:
                    # Check the first point (assumption: group belongs to same component)
                    if layer2_component.covers(points[pt_indices[0]]):
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
        print(mudguards)
        print(overlays)

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
                cv2.line(vis, tuple(coords[i]), tuple(coords[i + 1]), color, thickness)
        return vis
    
    @staticmethod
    def visualize_stitching_error(vis, connected_points, neighbour_margin, boundary_margin, max_line_dist, type='points'):
        """
        Visualize stitching point spacing and boundary deviation.
        """
        stitching_alignment_candidates = []

        for ring_info, processed_pts in connected_points.items():
            pts = np.array([[pt["point"].x, pt["point"].y] for pt in processed_pts], dtype=np.float32)
            projected_pts = np.array([[pt["projected_point"].x, pt["projected_point"].y] for pt in processed_pts], dtype=np.float32)
            distances = np.array([pt["distance"] for pt in processed_pts])
            neighbour_dist, med_neighbour_dist = StitchingProcessor.compute_distance_along_connections(processed_pts)

            num_connected_lines = np.zeros(len(processed_pts), dtype=np.int32)
            med_boundary_dist = np.median(distances)

            # ---- Connections ----
            p1s = pts[:-1]
            p2s = pts[1:]
            neighbour_dist = np.array(neighbour_dist)

            mask_connected = neighbour_dist < max_line_dist
            num_connected_lines[:-1] += mask_connected.astype(int)
            num_connected_lines[1:] += mask_connected.astype(int)

            # Batch draw connected lines
            for i in np.where(mask_connected)[0]:
                spi_ok = abs(neighbour_dist[i] - med_neighbour_dist) < neighbour_margin
                color = (255, 0, 0) if spi_ok else (0, 0, 255)
                cv2.line(vis, tuple(p1s[i].astype(int)), tuple(p2s[i].astype(int)), color, 1)

            # Candidate stitching misalignment
            if type == 'points' and ring_info[1] == "layer2":
                mask_candidate = (neighbour_dist < max_line_dist * 5.3) & (~mask_connected)
                for i in np.where(mask_candidate)[0]:
                    stitching_alignment_candidates.append(LineString([Point(*p1s[i]), Point(*p2s[i])]))

            # ---- Points & projections ----
            margin_ok = (np.abs(distances - med_boundary_dist) < boundary_margin) | (num_connected_lines < 2)
            for i in range(len(pts)):
                color = (255, 0, 0) if margin_ok[i] else (0, 0, 255)
                cv2.circle(vis, tuple(pts[i].astype(int)), 1, color, -1)

            # Projected points
            dists_proj = np.linalg.norm(pts - projected_pts, axis=1)
            mask_proj = dists_proj < max_line_dist
            for i in np.where(mask_proj)[0]:
                cv2.line(vis, tuple(pts[i].astype(int)), tuple(projected_pts[i].astype(int)), (0, 255, 255), 1)
                cv2.circle(vis, tuple(projected_pts[i].astype(int)), 1, (0, 0, 0), 1)
        return vis, stitching_alignment_candidates
    
    @staticmethod
    def visualize_component_alignment_error(vis, matches, max_component_offset_distance):
        for match in matches:
            alignment_ok = match["distance"] < max_component_offset_distance
            color = (255, 0, 0) if alignment_ok else (0, 0, 255)
            thickness = 1 if alignment_ok else 3

            # Overlay line (vectorized)
            overlay_pts = np.column_stack((
                match["overlay_line"]["x"],
                match["overlay_line"]["y"]
            )).astype(np.int32)

            if len(overlay_pts) > 1:
                cv2.polylines(vis,[overlay_pts],isClosed=False,color=color,thickness=thickness)

            # Mudguard line (vectorized)
            mudguard_pts = np.column_stack((
                match["mudguard_line"]["x"],
                match["mudguard_line"]["y"]
            )).astype(np.int32)

            if len(mudguard_pts) > 1:
                cv2.polylines(vis, [mudguard_pts], isClosed=False, color=(0, 0, 0), thickness=3)
        return vis
    
    @staticmethod
    def visualize_stitching_alignment_error(vis, stitching_alignment_to_check, max_stitching_offset_distance):
        """
        Highlight stitching segments exceeding allowed offset.
        """
        for line, (d1, d2) in stitching_alignment_to_check:
            if d1 > max_stitching_offset_distance or d2 > max_stitching_offset_distance:
                pts = np.asarray(line.coords, dtype=np.int32)
                cv2.polylines(vis, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
        return vis