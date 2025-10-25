CREATE TABLE IF NOT EXISTS f1_results_transformed (
    race_id SERIAL PRIMARY KEY,
    season INT,
    round INT,
    race_name VARCHAR(255),
    date DATE,
    winner VARCHAR(255),
    constructor VARCHAR(255),
    laps INT,
    time VARCHAR(50)
);
