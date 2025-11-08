# Formula 1 ML Pipeline - Core Star Schema

## Overview
The current ETL pipeline (`etl/extract_data.py`) reads **three CSV files** only:

- `data/races.csv`
- `data/results.csv`
- `data/drivers.csv`

Those sources already provide enough information to build a compact star schema that supports the classic analytics workflows and the existing `f1_results_transformed` table. This document captures that minimal model so it is easy to understand, populate, and evolve.

## Star Schema Diagram

```mermaid
erDiagram
FACT_RACE_RESULTS {
    bigint result_id PK
    int race_id FK
    int driver_id FK
    int constructor_id
    int grid_position
    int finish_position
    varchar position_text
    decimal points
}

DIM_RACE {
    int race_id PK
    int season_year
    int round_number
    varchar race_name
}

DIM_DRIVER {
    int driver_id PK
    varchar driver_ref
    int car_number
    varchar code
    varchar forename
    varchar surname
    date date_of_birth
    varchar nationality
}

FACT_RACE_RESULTS }o--|| DIM_RACE : race_id
FACT_RACE_RESULTS }o--|| DIM_DRIVER : driver_id
```mermaid
erDiagram
    FACT_DRIVER_RACE_RESULT {
        int driverId
        int raceId
        int year
        int round
        int position
        float points
        int win
        float win_rate
        float avg_points
    }
    DIM_DRIVER {
        int driverId
        string surname
    }
    DIM_RACE {
        int raceId
        int year
        int round
        string name
    }
    FACT_DRIVER_RACE_RESULT ||--|{ DIM_DRIVER : "driverId"
    FACT_DRIVER_RACE_RESULT ||--|{ DIM_RACE : "raceId"
```

## Sample Queries

```sql
-- Points by driver and season
SELECT
    d.surname,
    r.season_year,
    SUM(fr.points) AS total_points
FROM FACT_RACE_RESULTS fr
JOIN DIM_DRIVER d ON fr.driver_id = d.driver_id
JOIN DIM_RACE r   ON fr.race_id = r.race_id
GROUP BY d.surname, r.season_year
ORDER BY total_points DESC;
```

```sql
-- Average finishing position per driver
SELECT
    d.surname,
    AVG(fr.finish_position) AS avg_finish
FROM FACT_RACE_RESULTS fr
JOIN DIM_DRIVER d ON fr.driver_id = d.driver_id
GROUP BY d.surname
ORDER BY avg_finish;
```

## Why This Works
- Mirrors the joins already implemented in `extract_data.py`
- Minimal number of tables keeps maintenance low and onboarding simple
- Provides a clean base for dashboards, notebooks, or additional ML feature engineering
- Easy to extend: add new columns or dimensions when more CSV sources are included in the ETL flow

---

**Last Updated**: 2025-11-04  
**Scope**: Current CSV-driven pipeline (`races.csv`, `results.csv`, `drivers.csv`)
        int number
