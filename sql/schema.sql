CREATE TABLE images(
    img_name VARCHAR(255) PRIMARY KEY NOT NULL,
    img_data LONGBLOB NOT NULL
);

CREATE TABLE slider_values(
    img_name VARCHAR(255) PRIMARY KEY NOT NULL,
    neighbour_margin_factor DECIMAL,
    boundary_margin_factor DECIMAL,
    max_connected_line_dist DECIMAL,
    max_component_offset_dist DECIMAL,
    max_stitching_offset_dist DECIMAL
);

CREATE TABLE polygons (
    img_name VARCHAR(255) NOT NULL,
    polygon_index INT,
    label VARCHAR(255),
    vertices JSON NULL,
    PRIMARY KEY (img_name, polygon_index)
);

CREATE TABLE connected_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    img_name VARCHAR(255) NOT NULL,
    boundary_linestring JSON NULL,
    layer VARCHAR(255) NULL,
    point_data JSON NULL,
    INDEX idx_img_name (img_name)
);

CREATE TABLE connected_inner_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    img_name VARCHAR(255) NOT NULL,
    boundary_linestring JSON NULL,
    layer VARCHAR(255) NULL,
    point_data JSON NULL,
    INDEX idx_img_name (img_name)
);

CREATE TABLE matches (
    img_name VARCHAR(255) PRIMARY KEY,
    match_data JSON NULL
);

CREATE TABLE stitching_alignment_closest_boundary (
    img_name VARCHAR(255) PRIMARY KEY,
    lines_json JSON NULL
);