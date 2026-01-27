CREATE TABLE images(
    img_path VARCHAR(255) PRIMARY KEY NOT NULL,
    xml_path VARCHAR(255) NOT NULL,
    has_save BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE slider_values(
    img_path VARCHAR(255) PRIMARY KEY NOT NULL,
    buffer_distance DECIMAL,
    neighbour_margin_factor DECIMAL,
    boundary_margin_factor DECIMAL,
    max_connected_line_dist DECIMAL,
    max_component_offset_distance DECIMAL,
    max_stitching_offset_distance DECIMAL
);

CREATE TABLE polygons (
    img_path VARCHAR(255) NOT NULL,
    polygon_index INT NOT NULL,
    label VARCHAR(255),
    vertices JSON NOT NULL,
    PRIMARY KEY (img_path, polygon_index),
    INDEX idx_img_path (img_path)
);

CREATE TABLE connected_points (
    img_path VARCHAR(255) NOT NULL,
    boundary_linestring JSON NOT NULL,
    point_index INT NOT NULL,         
    point_x DECIMAL(10, 6) NOT NULL,
    point_y DECIMAL(10, 6) NOT NULL,
    sorting_distance DECIMAL(10, 6) NOT NULL,
    projected_point_x DECIMAL(10, 6),
    projected_point_y DECIMAL(10, 6),
    distance DECIMAL(10, 6),
    PRIMARY KEY (img_path, boundary_linestring, point_index),
    INDEX idx_img_poly (img_path, polygon_index)
);

CREATE TABLE inner_connected_points (
    img_path VARCHAR(255) NOT NULL,
    boundary_linestring JSON NOT NULL,
    point_index INT NOT NULL,        
    point_x DECIMAL(10, 6) NOT NULL,
    point_y DECIMAL(10, 6) NOT NULL,
    sorting_distance DECIMAL(10, 6) NOT NULL,
    projected_point_x DECIMAL(10, 6),
    projected_point_y DECIMAL(10, 6),
    distance DECIMAL(10, 6),
    PRIMARY KEY (img_path, boundary_linestring, point_index),
    INDEX idx_img_poly (img_path, polygon_index)
);