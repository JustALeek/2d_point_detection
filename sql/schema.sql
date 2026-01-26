CREATE TABLE images(
    image_id INT PRIMARY KEY AUTO_INCREMENT,
    orig_img_path VARCHAR(255) NOT NULL,
    xml_path VARCHAR(255) NOT NULL
);

CREATE TABLE slider_variables(
    image_id INT PRIMARY KEY AUTO_INCREMENT,
    buffer_distance DECIMAL,
    neighbour_margin_factor DECIMAL,
    boundary_margin_factor DECIMAL,
    max_connected_line_dist DECIMAL,
    max_component_offset_distance DECIMAL,
    max_stitching_offset_distance DECIMAL
);

CREATE TABLE polygons(
    image_id INT,
    polygon_index INT,
    layer VARCHAR,
    PRIMARY KEY (image_id, polygon_index)
);

CREATE TABLE vertices(
    image_id INT,
    polygon_index INT,
    vertex_x DECIMAL,
    vertex_y DECIMAL,
    PRIMARY KEY (image_id, polygon_index)
);

CREATE TABLE connected_points(
    image_id INT,
    polygon_index INT,
    point_x DECIMAL,
    point_y DECIMAL,
    sorting_distance DECIMAL,
    projected_point_x DECIMAL,
    projected_point_y DECIMAL,
    distance DECIMAL,
    PRIMARY KEY (image_id, polygon_index)
);

CREATE TABLE inner_connected_points(
    image_id INT,
    polygon_index INT,
    point_x DECIMAL,
    point_y DECIMAL,
    sorting_distance DECIMAL,
    projected_point_x DECIMAL,
    projected_point_y DECIMAL,
    distance DECIMAL,
    PRIMARY KEY (image_id, polygon_index)
);