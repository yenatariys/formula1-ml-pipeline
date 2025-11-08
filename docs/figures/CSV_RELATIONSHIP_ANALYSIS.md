# CSV Relationship Graph Analysis

## Overview

This analysis explores the relationships between CSV files in the Formula 1 ETL pipeline. It helps understand the data model structure, dependencies, and flow of data through the system.

## Generated Files

1. **csv_relationship_graph.png** - Visual representation of table relationships
2. **csv_dependency_report.txt** - Detailed text report of dependencies
3. **csv_schema.json** - Machine-readable schema with relationships

## Key Findings

### Core Tables (Most Referenced)
These tables are fundamental to the data model and are referenced by many other tables:

1. **results.csv** - 26,759 records, 1.7 MB
   - The central fact table containing race results
   - Referenced by 4 other tables

2. **races.csv** - 1,125 records, 161 KB  
   - Contains race event information
   - Links circuits, seasons, and results

3. **drivers.csv** - 861 records, 93 KB
   - Driver master data
   - Referenced by 6 different tables

4. **constructors.csv** - 212 records, 17 KB
   - Constructor (team) master data
   - Referenced by 5 different tables

### Most Complex Tables
These tables have the most dependencies on other tables:

1. **results.csv** - Depends on: races, drivers, constructors, status
2. **qualifying.csv** - Depends on: races, drivers, constructors
3. **sprint_results.csv** - Depends on: races, drivers, constructors

### Data Categories

#### Core Entities (5 tables)
- `drivers.csv` - 861 drivers
- `constructors.csv` - 212 constructors
- `circuits.csv` - 77 circuits
- `races.csv` - 1,125 races
- `seasons.csv` - 75 seasons

#### Race Results (3 tables)
- `results.csv` - 26,759 race results
- `sprint_results.csv` - 360 sprint results
- `qualifying.csv` - 10,494 qualifying results

#### Standings (2 tables)
- `driver_standings.csv` - 34,863 standing records
- `constructor_standings.csv` - 13,391 standing records

#### Race Details (2 tables)
- `lap_times.csv` - 589,081 lap times (17.8 MB - largest file)
- `pit_stops.csv` - 11,371 pit stop records

#### Reference (2 tables)
- `status.csv` - 139 finish statuses
- `constructor_results.csv` - 12,625 constructor results

#### Transformed (1 table)
- `f1_results_joined.csv` - 26,759 denormalized results (output of ETL)

## Data Flow Paths

### From Core Entities to Results

1. **drivers.csv → results.csv** (Direct)
2. **constructors.csv → results.csv** (Direct)
3. **circuits.csv → races.csv → results.csv** (2-step)
4. **seasons.csv → races.csv → results.csv** (2-step)

## Database Statistics

- **Total CSV files:** 15
- **Total relationships:** 22 foreign key relationships
- **Total data size:** 23.5 MB
- **Total records:** 728,192 rows across all tables

## Usage

Run the analysis script:

```bash
python analytics/csv_relationship_graph.py
```

This will generate:
- A visual graph showing table relationships
- A detailed dependency report
- A JSON schema for programmatic access

## Insights for ETL Design

1. **results.csv is the central fact table** - Most queries and joins will involve this table
2. **races.csv is a critical dimension** - Links temporal (seasons) and spatial (circuits) context
3. **Master data tables** (drivers, constructors, circuits) are heavily reused
4. **lap_times.csv is the largest** - Consider partitioning strategies for performance
5. **Clear star schema pattern** - Fact table (results) surrounded by dimension tables

## Next Steps

1. Consider indexing strategies for high-traffic foreign keys
2. Optimize joins between results and master tables
3. Partition large tables (lap_times) by year or race
4. Create materialized views for common join patterns
5. Monitor query performance on multi-table joins
