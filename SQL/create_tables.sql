CREATE DATABASE IF NOT EXISTS seismic_data;
USE seismic_data;

DROP TABLE IF EXISTS earthquakes;

CREATE TABLE earthquakes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    time DATETIME,
    latitude FLOAT,
    longitude FLOAT,
    depth FLOAT,
    mag FLOAT,
    magtype VARCHAR(10),
    net VARCHAR(10),
    place TEXT,
    type VARCHAR(50),
    status VARCHAR(20),
    locationsource VARCHAR(10),
    magsource VARCHAR(10)
);


USE seismic_data;
SELECT COUNT(*) FROM earthquakes;
