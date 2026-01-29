CREATE TABLE images(
    img_path VARCHAR(255) PRIMARY KEY NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE slider_values(
    img_path VARCHAR(255) PRIMARY KEY NOT NULL,
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
    PRIMARY KEY (img_path, polygon_index)
);

CREATE TABLE connected_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    img_path VARCHAR(255) NOT NULL,
    boundary_linestring JSON NOT NULL,
    layer VARCHAR(255) NOT NULL,
    point_data JSON NOT NULL,
    INDEX idx_img_path (img_path)
);

CREATE TABLE connected_inner_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    img_path VARCHAR(255) NOT NULL,
    boundary_linestring JSON NOT NULL,
    layer VARCHAR(255) NOT NULL,
    point_data JSON NOT NULL,
    INDEX idx_img_path (img_path)
);

CREATE TABLE matches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    img_path VARCHAR(255) NOT NULL,
    match_data JSON NOT NULL,
    INDEX idx_img_path (img_path)
);

CREATE TABLE stitching_alignment_closest_boundary (
    img_path VARCHAR(255) PRIMARY KEY NOT NULL,
    line_data TEXT NOT NULL
);