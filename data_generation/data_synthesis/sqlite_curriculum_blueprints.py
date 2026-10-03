"""Calibrated post-analysis SQLite samples for the Spider 2.0 environment.

These tasks were authored after auditing the Qwen3.5-9B/strong-teacher gap and
the rejected ``hard_pilot_10``.  They deliberately cover a mixture of stable
wins, the current learning frontier, bounded multi-stage problems, and one
stretch item.  Every mapping is still checked against the offline catalog,
live database, controlled mutations, and the Spider reference complexity
profile by :mod:`data_synthesis.pipeline`.
"""

from __future__ import annotations

from typing import Any

from .curriculum import difficulty_contract
from .sqlite_pilot_blueprints import mutation


# Aggregate quantiles from 106 officially result-verified Spider 2.0 SQLite
# predictions.  No reference SQL text or benchmark question is included here.
REFERENCE_PROFILE: dict[str, Any] = {
    "reference_kind": "official-result-verified local predictions; aggregate complexity use only",
    "distributions": {
        "structural_score": {
            "p25": 11.394,
            "median": 16.834,
            "p75": 22.833,
            "p90": 32.782,
        },
        "sql_tokens": {
            "p25": 127.0,
            "median": 170.0,
            "p75": 231.5,
            "p90": 304.5,
        },
    },
}


def target(band: str) -> dict[str, Any]:
    return difficulty_contract(REFERENCE_PROFILE, band)


CURRICULUM_PILOT_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "sample_id": "sqlite_curriculum_001",
        "database_id": "chinook",
        "difficulty_band": "foundation",
        "template_id": "bounded_single_table_group_leaderboard",
        "instruction": (
            "Build a customer-country leaderboard from all Chinook customers whose country is not null. "
            "Return exactly the five countries with the most customers. The output columns must be named "
            "`customer_country` and `customer_count`. Rank larger customer_count first and break ties by "
            "customer_country alphabetically."
        ),
        "sql": """SELECT Country AS customer_country,
       COUNT(*) AS customer_count
FROM customers
WHERE Country IS NOT NULL
GROUP BY Country
ORDER BY customer_count DESC, customer_country ASC
LIMIT 5""",
        "expected_columns": ["customer_country", "customer_count"],
        "row_bounds": [5, 5],
        "operators": ["single_table", "group_by", "top_k", "deterministic_tie_break"],
        "roles": [
            {
                "role": "customer_fact",
                "kind": "table",
                "query": "customer country location",
                "selected": "customers",
            },
            {
                "role": "country_dimension",
                "kind": "column",
                "table": "customers",
                "query": "customer country",
                "selected": "Country",
                "type": "text",
            },
        ],
        "joins": [],
        "mutations": [
            mutation(
                "reverse_country_leaderboard",
                "sorting",
                "ORDER BY customer_count DESC",
                "ORDER BY customer_count ASC",
            ),
            mutation("return_six_countries", "top_k", "LIMIT 5", "LIMIT 6"),
            mutation(
                "usa_only_population",
                "filter",
                "Country IS NOT NULL",
                "Country = 'USA'",
            ),
        ],
        "complexity_target": target("foundation"),
        "semantic_risks": ["deterministic_top_k"],
    },
    {
        "sample_id": "sqlite_curriculum_002",
        "database_id": "northwind",
        "difficulty_band": "core",
        "template_id": "period_workload_rank_with_late_count",
        "instruction": (
            "For Northwind orders placed during calendar year 1997, rank employees by the number of orders "
            "they handled. A late shipment is an order whose shipped date is after its required date. Keep all "
            "employees whose dense workload rank is 1 through 5. Return `employee_name`, `order_count`, "
            "`total_freight` rounded to two decimals, `late_shipments`, and `workload_rank`. Order by "
            "workload_rank ascending and then employee_name alphabetically."
        ),
        "sql": """WITH employee_load AS (
    SELECT e.EmployeeId,
           e.FirstName || ' ' || e.LastName AS employee_name,
           COUNT(*) AS order_count,
           ROUND(SUM(o.Freight), 2) AS total_freight,
           SUM(CASE WHEN o.ShippedDate > o.RequiredDate THEN 1 ELSE 0 END) AS late_shipments
    FROM orders AS o
    JOIN employees AS e ON e.EmployeeId = o.EmployeeId
    WHERE o.OrderDate >= '1997-01-01' AND o.OrderDate < '1998-01-01'
    GROUP BY e.EmployeeId, e.FirstName, e.LastName
), ranked AS (
    SELECT employee_name,
           order_count,
           total_freight,
           late_shipments,
           DENSE_RANK() OVER (ORDER BY order_count DESC) AS workload_rank
    FROM employee_load
)
SELECT employee_name,
       order_count,
       total_freight,
       late_shipments,
       workload_rank
FROM ranked
WHERE workload_rank <= 5
ORDER BY workload_rank ASC, employee_name ASC""",
        "expected_columns": [
            "employee_name",
            "order_count",
            "total_freight",
            "late_shipments",
            "workload_rank",
        ],
        "row_bounds": [5, 9],
        "operators": [
            "date_range",
            "join",
            "conditional_aggregate",
            "dense_rank",
            "tie_inclusive_top_k",
        ],
        "roles": [
            {
                "role": "order_fact",
                "kind": "table",
                "query": "orders employee order date required shipped freight",
                "selected": "orders",
            },
            {
                "role": "employee_dimension",
                "kind": "table",
                "query": "employee first last name",
                "selected": "employees",
            },
            {
                "role": "freight_measure",
                "kind": "column",
                "table": "orders",
                "query": "freight amount",
                "selected": "freight",
                "type": "numeric",
            },
        ],
        "joins": [["orders", "employeeid", "employees", "employeeid"]],
        "mutations": [
            mutation(
                "wrong_order_year",
                "filter",
                "o.OrderDate >= '1997-01-01' AND o.OrderDate < '1998-01-01'",
                "o.OrderDate >= '1998-01-01' AND o.OrderDate < '1999-01-01'",
            ),
            mutation(
                "on_time_instead_of_late",
                "predicate",
                "o.ShippedDate > o.RequiredDate",
                "o.ShippedDate <= o.RequiredDate",
            ),
            mutation(
                "rank_lightest_workload",
                "sorting",
                "DENSE_RANK() OVER (ORDER BY order_count DESC)",
                "DENSE_RANK() OVER (ORDER BY order_count ASC)",
            ),
        ],
        "complexity_target": target("core"),
        "semantic_risks": ["calendar_boundary", "conditional_event_definition", "tie_inclusive_rank"],
    },
    {
        "sample_id": "sqlite_curriculum_003",
        "database_id": "f1",
        "difficulty_band": "growth",
        "template_id": "qualified_multi_season_driver_rank",
        "instruction": (
            "Measure Formula 1 driver consistency over seasons 2010 through 2020 inclusive. First summarize "
            "each driver-season, counting races entered, points, and wins where finishing position equals 1. "
            "A season qualifies only when the driver entered at least 10 distinct races; a driver qualifies "
            "only with at least five such seasons. Rank qualified drivers by average points per qualifying "
            "season using dense rank and keep ranks 1 through 10. Return `driver_name`, `seasons_entered`, "
            "`avg_points_per_season` rounded to two decimals, `total_wins`, and `performance_rank`, ordered by "
            "performance_rank and then driver_name."
        ),
        "sql": """WITH season_driver AS (
    SELECT r.year,
           res.driver_id,
           COUNT(DISTINCT res.race_id) AS races_entered,
           ROUND(SUM(res.points), 1) AS season_points,
           SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS season_wins
    FROM results AS res
    JOIN races AS r ON r.race_id = res.race_id
    WHERE r.year BETWEEN 2010 AND 2020
    GROUP BY r.year, res.driver_id
), consistent_drivers AS (
    SELECT driver_id,
           COUNT(*) AS seasons_entered,
           ROUND(AVG(season_points), 2) AS avg_points_per_season,
           SUM(season_wins) AS total_wins
    FROM season_driver
    WHERE races_entered >= 10
    GROUP BY driver_id
    HAVING COUNT(*) >= 5
), ranked AS (
    SELECT d.forename || ' ' || d.surname AS driver_name,
           cd.seasons_entered,
           cd.avg_points_per_season,
           cd.total_wins,
           DENSE_RANK() OVER (ORDER BY cd.avg_points_per_season DESC) AS performance_rank
    FROM consistent_drivers AS cd
    JOIN drivers AS d ON d.driver_id = cd.driver_id
)
SELECT driver_name,
       seasons_entered,
       avg_points_per_season,
       total_wins,
       performance_rank
FROM ranked
WHERE performance_rank <= 10
ORDER BY performance_rank ASC, driver_name ASC""",
        "expected_columns": [
            "driver_name",
            "seasons_entered",
            "avg_points_per_season",
            "total_wins",
            "performance_rank",
        ],
        "row_bounds": [10, 20],
        "operators": [
            "multi_stage_cte",
            "two_grain_aggregation",
            "conditional_aggregate",
            "qualification",
            "dense_rank",
        ],
        "roles": [
            {
                "role": "race_result_fact",
                "kind": "table",
                "query": "race driver finish position points",
                "selected": "results",
            },
            {
                "role": "race_season_dimension",
                "kind": "table",
                "query": "race season year",
                "selected": "races",
            },
            {
                "role": "driver_dimension",
                "kind": "table",
                "query": "driver first surname name",
                "selected": "drivers",
            },
            {
                "role": "points_measure",
                "kind": "column",
                "table": "results",
                "query": "championship points",
                "selected": "points",
                "type": "numeric",
            },
        ],
        "joins": [
            ["results", "race_id", "races", "race_id"],
            ["results", "driver_id", "drivers", "driver_id"],
        ],
        "mutations": [
            mutation(
                "wrong_season_window",
                "filter",
                "r.year BETWEEN 2010 AND 2020",
                "r.year BETWEEN 2000 AND 2010",
            ),
            mutation(
                "stricter_race_qualification",
                "threshold",
                "WHERE races_entered >= 10",
                "WHERE races_entered >= 15",
            ),
            mutation(
                "rank_lowest_average_points",
                "sorting",
                "DENSE_RANK() OVER (ORDER BY cd.avg_points_per_season DESC)",
                "DENSE_RANK() OVER (ORDER BY cd.avg_points_per_season ASC)",
            ),
        ],
        "complexity_target": target("growth"),
        "semantic_risks": ["two_stage_grain", "qualification_before_average", "tie_inclusive_rank"],
    },
    {
        "sample_id": "sqlite_curriculum_004",
        "database_id": "Pagila",
        "difficulty_band": "stretch",
        "template_id": "store_to_category_revenue_distribution",
        "instruction": (
            "Analyze Pagila payment revenue by film category without duplicating payment rows. First aggregate "
            "rental count and payment revenue separately for each store-category pair, then summarize categories "
            "across stores. Rank categories by total revenue using dense rank and keep ranks 1 through 5, including "
            "ties. Return `category`, `rental_count`, `stores_with_rentals`, `revenue`, `lowest_store_revenue`, "
            "`highest_store_revenue`, `revenue_share_pct`, `revenue_rank`, `gap_from_previous_category` defined as "
            "current revenue minus the previous higher-ranked category's revenue (null for the first), and "
            "`cumulative_revenue_pct`. Round all money, shares, gaps, and cumulative percentages to two decimals. "
            "Order by revenue_rank and category."
        ),
        "sql": """WITH store_category AS (
    SELECT i.store_id,
           c.name AS category,
           COUNT(DISTINCT r.rental_id) AS rental_count,
           ROUND(SUM(p.amount), 2) AS revenue
    FROM payment AS p
    JOIN rental AS r ON r.rental_id = p.rental_id
    JOIN inventory AS i ON i.inventory_id = r.inventory_id
    JOIN film_category AS fc ON fc.film_id = i.film_id
    JOIN category AS c ON c.category_id = fc.category_id
    GROUP BY i.store_id, c.category_id, c.name
), category_summary AS (
    SELECT category,
           SUM(rental_count) AS rental_count,
           COUNT(*) AS stores_with_rentals,
           ROUND(SUM(revenue), 2) AS revenue,
           ROUND(MIN(revenue), 2) AS lowest_store_revenue,
           ROUND(MAX(revenue), 2) AS highest_store_revenue
    FROM store_category
    GROUP BY category
), ranked AS (
    SELECT category,
           rental_count,
           stores_with_rentals,
           revenue,
           lowest_store_revenue,
           highest_store_revenue,
           ROUND(100.0 * revenue / NULLIF(SUM(revenue) OVER (), 0), 2) AS revenue_share_pct,
           DENSE_RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
           ROUND(revenue - LAG(revenue) OVER (ORDER BY revenue DESC), 2) AS gap_from_previous_category,
           ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) /
                 NULLIF(SUM(revenue) OVER (), 0), 2) AS cumulative_revenue_pct
    FROM category_summary
)
SELECT category,
       rental_count,
       stores_with_rentals,
       revenue,
       lowest_store_revenue,
       highest_store_revenue,
       revenue_share_pct,
       revenue_rank,
       gap_from_previous_category,
       cumulative_revenue_pct
FROM ranked
WHERE revenue_rank <= 5
ORDER BY revenue_rank ASC, category ASC""",
        "expected_columns": [
            "category",
            "rental_count",
            "stores_with_rentals",
            "revenue",
            "lowest_store_revenue",
            "highest_store_revenue",
            "revenue_share_pct",
            "revenue_rank",
            "gap_from_previous_category",
            "cumulative_revenue_pct",
        ],
        "row_bounds": [5, 8],
        "operators": [
            "multi_stage_cte",
            "grain_safe_multi_fact",
            "window_share",
            "dense_rank",
            "lag",
            "cumulative_window",
        ],
        "roles": [
            {
                "role": "payment_fact",
                "kind": "table",
                "query": "payment rental amount revenue",
                "selected": "payment",
            },
            {
                "role": "rental_fact",
                "kind": "table",
                "query": "rental inventory store",
                "selected": "rental",
            },
            {
                "role": "inventory_bridge",
                "kind": "table",
                "query": "inventory film store",
                "selected": "inventory",
            },
            {
                "role": "film_category_bridge",
                "kind": "table",
                "query": "film category mapping",
                "selected": "film_category",
            },
            {
                "role": "category_dimension",
                "kind": "table",
                "query": "category name",
                "selected": "category",
            },
            {
                "role": "payment_amount",
                "kind": "column",
                "table": "payment",
                "query": "payment amount revenue",
                "selected": "amount",
                "type": "numeric",
            },
        ],
        "joins": [
            ["payment", "rental_id", "rental", "rental_id"],
            ["rental", "inventory_id", "inventory", "inventory_id"],
            ["inventory", "film_id", "film_category", "film_id"],
            ["film_category", "category_id", "category", "category_id"],
        ],
        "mutations": [
            mutation(
                "average_instead_of_total_revenue",
                "aggregation",
                "ROUND(SUM(p.amount), 2) AS revenue",
                "ROUND(AVG(p.amount), 2) AS revenue",
            ),
            mutation(
                "average_store_rentals",
                "aggregation",
                "SUM(rental_count) AS rental_count",
                "AVG(rental_count) AS rental_count",
            ),
            mutation(
                "include_sixth_revenue_rank",
                "threshold",
                "WHERE revenue_rank <= 5",
                "WHERE revenue_rank <= 6",
            ),
        ],
        "complexity_target": target("stretch"),
        "semantic_risks": [
            "payment_grain",
            "store_then_category_aggregation",
            "window_ordering",
            "tie_inclusive_rank",
        ],
    },
    {
        "sample_id": "sqlite_curriculum_005",
        "database_id": "California_Traffic_Collision",
        "difficulty_band": "foundation",
        "template_id": "filtered_single_table_category_summary",
        "instruction": (
            "For collisions dated from 2021-01-01 through 2021-12-31, keep rows whose primary weather "
            "condition is not null. Return the five most common conditions with output columns "
            "`weather_condition`, `collision_count`, and total `killed_victims`. Sort by collision_count "
            "descending and then weather_condition alphabetically."
        ),
        "sql": """SELECT weather_1 AS weather_condition,
       COUNT(*) AS collision_count,
       SUM(killed_victims) AS killed_victims
FROM collisions
WHERE collision_date >= '2021-01-01' AND collision_date < '2022-01-01'
  AND weather_1 IS NOT NULL
GROUP BY weather_1
ORDER BY collision_count DESC, weather_condition ASC
LIMIT 5""",
        "expected_columns": ["weather_condition", "collision_count", "killed_victims"],
        "row_bounds": [5, 5],
        "operators": ["single_table", "date_range", "group_by", "top_k"],
        "roles": [
            {
                "role": "collision_fact",
                "kind": "table",
                "query": "collision date weather killed victims",
                "selected": "collisions",
            },
            {
                "role": "weather_dimension",
                "kind": "column",
                "table": "collisions",
                "query": "primary weather condition",
                "selected": "weather_1",
                "type": "text",
            },
            {
                "role": "fatality_measure",
                "kind": "column",
                "table": "collisions",
                "query": "killed victims",
                "selected": "killed_victims",
                "type": "numeric",
            },
        ],
        "joins": [],
        "mutations": [
            mutation(
                "wrong_collision_year",
                "filter",
                "collision_date >= '2021-01-01' AND collision_date < '2022-01-01'",
                "collision_date >= '2020-01-01' AND collision_date < '2021-01-01'",
            ),
            mutation(
                "reverse_weather_frequency",
                "sorting",
                "ORDER BY collision_count DESC",
                "ORDER BY collision_count ASC",
            ),
            mutation(
                "injured_instead_of_killed",
                "measure",
                "SUM(killed_victims) AS killed_victims",
                "SUM(injured_victims) AS killed_victims",
            ),
        ],
        "complexity_target": target("foundation"),
        "semantic_risks": ["half_open_date_boundary", "top_k_tie_break"],
    },
    {
        "sample_id": "sqlite_curriculum_006",
        "database_id": "sqlite-sakila",
        "difficulty_band": "growth",
        "template_id": "category_rating_vs_category_baseline",
        "instruction": (
            "For each Sakila film category, summarize every rating represented in that category, then select "
            "the rating or tied ratings with the highest average rental rate in that category. For each selected "
            "pair return `category`, `film_rating`, `film_count`, `avg_rental_rate`, "
            "`category_avg_rental_rate` defined as the unweighted average of the rating-level averages in that "
            "category, `rental_rate_vs_category_avg`, and `high_replacement_cost_pct`, the percentage of films "
            "whose replacement cost is at least 20. Round rates, differences, and percentages to two decimals. "
            "Order by rental_rate_vs_category_avg descending and category alphabetically."
        ),
        "sql": """WITH category_rating AS (
    SELECT c.name AS category,
           f.rating AS film_rating,
           COUNT(*) AS film_count,
           ROUND(AVG(f.rental_rate), 2) AS avg_rental_rate,
           ROUND(100.0 * SUM(CASE WHEN f.replacement_cost >= 20 THEN 1 ELSE 0 END) / COUNT(*), 2) AS high_replacement_cost_pct
    FROM film AS f
    JOIN film_category AS fc ON fc.film_id = f.film_id
    JOIN category AS c ON c.category_id = fc.category_id
    WHERE f.rating IS NOT NULL
    GROUP BY c.category_id, c.name, f.rating
), compared AS (
    SELECT category,
           film_rating,
           film_count,
           avg_rental_rate,
           high_replacement_cost_pct,
           ROUND(AVG(avg_rental_rate) OVER (PARTITION BY category), 2) AS category_avg_rental_rate,
           DENSE_RANK() OVER (PARTITION BY category ORDER BY avg_rental_rate DESC) AS rating_rank_in_category
    FROM category_rating
), selected AS (
    SELECT category,
           film_rating,
           film_count,
           avg_rental_rate,
           category_avg_rental_rate,
           ROUND(avg_rental_rate - category_avg_rental_rate, 2) AS rental_rate_vs_category_avg,
           high_replacement_cost_pct,
           rating_rank_in_category
    FROM compared
    WHERE rating_rank_in_category = 1
)
SELECT category,
       film_rating,
       film_count,
       avg_rental_rate,
       category_avg_rental_rate,
       rental_rate_vs_category_avg,
       high_replacement_cost_pct
FROM selected
ORDER BY rental_rate_vs_category_avg DESC, category ASC""",
        "expected_columns": [
            "category",
            "film_rating",
            "film_count",
            "avg_rental_rate",
            "category_avg_rental_rate",
            "rental_rate_vs_category_avg",
            "high_replacement_cost_pct",
        ],
        "row_bounds": [16, 30],
        "operators": [
            "multi_stage_cte",
            "conditional_aggregate",
            "partitioned_window",
            "within_group_dense_rank",
        ],
        "roles": [
            {
                "role": "film_fact",
                "kind": "table",
                "query": "film rating rental rate replacement cost",
                "selected": "film",
            },
            {
                "role": "film_category_bridge",
                "kind": "table",
                "query": "film category mapping",
                "selected": "film_category",
            },
            {
                "role": "category_dimension",
                "kind": "table",
                "query": "category name",
                "selected": "category",
            },
            {
                "role": "rental_rate_measure",
                "kind": "column",
                "table": "film",
                "query": "rental rate price",
                "selected": "rental_rate",
                "type": "numeric",
            },
        ],
        "joins": [
            ["film", "film_id", "film_category", "film_id"],
            ["film_category", "category_id", "category", "category_id"],
        ],
        "mutations": [
            mutation(
                "higher_replacement_threshold",
                "threshold",
                "f.replacement_cost >= 20",
                "f.replacement_cost >= 25",
            ),
            mutation(
                "lowest_rate_per_category",
                "sorting",
                "PARTITION BY category ORDER BY avg_rental_rate DESC",
                "PARTITION BY category ORDER BY avg_rental_rate ASC",
            ),
            mutation(
                "top_two_ratings_per_category",
                "threshold",
                "WHERE rating_rank_in_category = 1",
                "WHERE rating_rank_in_category <= 2",
            ),
        ],
        "complexity_target": target("growth"),
        "semantic_risks": ["unweighted_group_baseline", "within_group_ties", "aggregate_then_window"],
    },
    {
        "sample_id": "sqlite_curriculum_007",
        "database_id": "Brazilian_E_Commerce",
        "difficulty_band": "core",
        "template_id": "state_delivery_latency_and_lateness",
        "instruction": (
            "For delivered marketplace orders purchased during calendar year 2018, summarize customer states "
            "with at least 100 delivered orders. Count each order once. Return `customer_state`, "
            "`delivered_orders`, distinct `unique_customers`, `avg_delivery_days` from purchase timestamp to "
            "customer delivery rounded to two decimals, and `late_delivery_pct`, the percentage delivered after "
            "the estimated delivery timestamp rounded to two decimals. Order by delivered_orders descending and "
            "customer_state alphabetically."
        ),
        "sql": """WITH state_delivery AS (
    SELECT c.customer_state,
           COUNT(DISTINCT o.order_id) AS delivered_orders,
           COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
           ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)), 2) AS avg_delivery_days,
           ROUND(100.0 * SUM(CASE WHEN datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date) THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_delivery_pct
    FROM olist_orders AS o
    JOIN olist_customers AS c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp >= '2018-01-01'
      AND o.order_purchase_timestamp < '2019-01-01'
    GROUP BY c.customer_state
)
SELECT customer_state,
       delivered_orders,
       unique_customers,
       avg_delivery_days,
       late_delivery_pct
FROM state_delivery
WHERE delivered_orders >= 100
ORDER BY delivered_orders DESC, customer_state ASC""",
        "expected_columns": [
            "customer_state",
            "delivered_orders",
            "unique_customers",
            "avg_delivery_days",
            "late_delivery_pct",
        ],
        "row_bounds": [15, 30],
        "operators": [
            "date_range",
            "join",
            "distinct_grain",
            "timestamp_delta",
            "conditional_percentage",
        ],
        "roles": [
            {
                "role": "order_fact",
                "kind": "table",
                "query": "order status purchase delivered estimated timestamp customer",
                "selected": "olist_orders",
            },
            {
                "role": "customer_dimension",
                "kind": "table",
                "query": "customer unique state",
                "selected": "olist_customers",
            },
            {
                "role": "delivery_timestamp",
                "kind": "column",
                "table": "olist_orders",
                "query": "customer delivered date",
                "selected": "order_delivered_customer_date",
                "type": "text",
            },
        ],
        "joins": [["olist_orders", "customer_id", "olist_customers", "customer_id"]],
        "mutations": [
            mutation(
                "wrong_purchase_year",
                "filter",
                "o.order_purchase_timestamp >= '2018-01-01'\n      AND o.order_purchase_timestamp < '2019-01-01'",
                "o.order_purchase_timestamp >= '2017-01-01'\n      AND o.order_purchase_timestamp < '2018-01-01'",
            ),
            mutation(
                "early_or_on_time_instead_of_late",
                "predicate",
                "datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date)",
                "datetime(o.order_delivered_customer_date) <= datetime(o.order_estimated_delivery_date)",
            ),
            mutation(
                "reverse_state_volume",
                "sorting",
                "ORDER BY delivered_orders DESC",
                "ORDER BY delivered_orders ASC",
            ),
        ],
        "complexity_target": target("core"),
        "semantic_risks": ["order_grain", "half_open_date_boundary", "timestamp_comparison"],
    },
    {
        "sample_id": "sqlite_curriculum_008",
        "database_id": "Airlines",
        "difficulty_band": "core",
        "template_id": "aircraft_arrival_delay_summary",
        "instruction": (
            "For flights marked Arrived with non-missing actual and scheduled arrival timestamps, compare "
            "aircraft models having at least 100 such flights. Treat the literal `\\N` as missing. Delay is "
            "actual arrival minus scheduled arrival in minutes. Return `aircraft_model` using the English JSON "
            "name, `arrived_flights`, `avg_arrival_delay_minutes`, `late_arrival_pct` for delays greater than zero, "
            "and `worst_delay_minutes`; round the three delay or percentage metrics to two decimals. Order by "
            "avg_arrival_delay_minutes descending and aircraft_model alphabetically."
        ),
        "sql": """WITH arrived AS (
    SELECT f.aircraft_code,
           (julianday(substr(f.actual_arrival, 1, 19)) -
            julianday(substr(f.scheduled_arrival, 1, 19))) * 1440.0 AS delay_minutes
    FROM flights AS f
    WHERE f.status = 'Arrived'
      AND f.actual_arrival <> '\\N'
      AND f.scheduled_arrival <> '\\N'
)
SELECT json_extract(a.model, '$.en') AS aircraft_model,
       COUNT(*) AS arrived_flights,
       ROUND(AVG(ar.delay_minutes), 2) AS avg_arrival_delay_minutes,
       ROUND(100.0 * SUM(CASE WHEN ar.delay_minutes > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_arrival_pct,
       ROUND(MAX(ar.delay_minutes), 2) AS worst_delay_minutes
FROM arrived AS ar
JOIN aircrafts_data AS a ON a.aircraft_code = ar.aircraft_code
GROUP BY ar.aircraft_code, a.model
HAVING COUNT(*) >= 100
ORDER BY avg_arrival_delay_minutes DESC, aircraft_model ASC""",
        "expected_columns": [
            "aircraft_model",
            "arrived_flights",
            "avg_arrival_delay_minutes",
            "late_arrival_pct",
            "worst_delay_minutes",
        ],
        "row_bounds": [5, 12],
        "operators": [
            "cte",
            "timestamp_normalization",
            "json_extract",
            "conditional_percentage",
            "group_by",
        ],
        "roles": [
            {
                "role": "flight_fact",
                "kind": "table",
                "query": "flight aircraft scheduled actual arrival status",
                "selected": "flights",
            },
            {
                "role": "aircraft_dimension",
                "kind": "table",
                "query": "aircraft code model range",
                "selected": "aircrafts_data",
            },
            {
                "role": "actual_arrival",
                "kind": "column",
                "table": "flights",
                "query": "actual arrival timestamp",
                "selected": "actual_arrival",
                "type": "temporal",
            },
        ],
        "joins": [["flights", "aircraft_code", "aircrafts_data", "aircraft_code"]],
        "mutations": [
            mutation(
                "departed_instead_of_arrived",
                "filter",
                "f.status = 'Arrived'",
                "f.status = 'Departed'",
            ),
            mutation(
                "include_zero_delay_as_late",
                "predicate",
                "ar.delay_minutes > 0",
                "ar.delay_minutes >= 0",
            ),
            mutation(
                "reverse_delay_leaderboard",
                "sorting",
                "ORDER BY avg_arrival_delay_minutes DESC",
                "ORDER BY avg_arrival_delay_minutes ASC",
            ),
        ],
        "complexity_target": target("core"),
        "semantic_risks": ["literal_missing_marker", "timestamp_normalization", "json_label"],
    },
    {
        "sample_id": "sqlite_curriculum_009",
        "database_id": "Baseball",
        "difficulty_band": "core",
        "template_id": "bounded_set_exclusion_rank",
        "instruction": (
            "Among players with All-Star records from 2010 through 2020 inclusive, exclude every player who has "
            "any row at all in the Hall of Fame table. Keep players with at least two All-Star selections, dense "
            "rank them by selection count, and retain all ranks 1 through 10 including ties. Return "
            "`player_name`, `all_star_selections`, `first_selection_year`, `last_selection_year`, and "
            "`selection_rank`, ordered by selection_rank and player_name."
        ),
        "sql": """WITH eligible_players AS (
    SELECT a.player_id,
           p.name_first || ' ' || p.name_last AS player_name,
           COUNT(*) AS all_star_selections,
           MIN(a.year) AS first_selection_year,
           MAX(a.year) AS last_selection_year
    FROM all_star AS a
    JOIN player AS p ON p.player_id = a.player_id
    WHERE a.year BETWEEN 2010 AND 2020
      AND NOT EXISTS (
          SELECT 1
          FROM hall_of_fame AS h
          WHERE h.player_id = a.player_id
      )
    GROUP BY a.player_id, p.name_first, p.name_last
    HAVING COUNT(*) >= 2
), ranked AS (
    SELECT player_name,
           all_star_selections,
           first_selection_year,
           last_selection_year,
           DENSE_RANK() OVER (ORDER BY all_star_selections DESC) AS selection_rank
    FROM eligible_players
)
SELECT player_name,
       all_star_selections,
       first_selection_year,
       last_selection_year,
       selection_rank
FROM ranked
WHERE selection_rank <= 10
ORDER BY selection_rank ASC, player_name ASC""",
        "expected_columns": [
            "player_name",
            "all_star_selections",
            "first_selection_year",
            "last_selection_year",
            "selection_rank",
        ],
        "row_bounds": [10, 150],
        "operators": ["join", "not_exists", "group_by", "having", "dense_rank"],
        "roles": [
            {
                "role": "all_star_fact",
                "kind": "table",
                "query": "player all star selection year",
                "selected": "all_star",
            },
            {
                "role": "player_dimension",
                "kind": "table",
                "query": "player first last name",
                "selected": "player",
            },
            {
                "role": "excluded_hall_population",
                "kind": "table",
                "query": "hall of fame player records",
                "selected": "hall_of_fame",
            },
        ],
        "joins": [["all_star", "player_id", "player", "player_id"]],
        "mutations": [
            mutation("include_only_hall_records", "set_exclusion", "AND NOT EXISTS (", "AND EXISTS ("),
            mutation(
                "wrong_all_star_decade",
                "filter",
                "a.year BETWEEN 2010 AND 2020",
                "a.year BETWEEN 2000 AND 2010",
            ),
            mutation(
                "rank_fewest_selections",
                "sorting",
                "DENSE_RANK() OVER (ORDER BY all_star_selections DESC)",
                "DENSE_RANK() OVER (ORDER BY all_star_selections ASC)",
            ),
        ],
        "complexity_target": target("core"),
        "semantic_risks": ["set_exclusion", "existence_vs_induction_status", "tie_inclusive_rank"],
    },
    {
        "sample_id": "sqlite_curriculum_010",
        "database_id": "northwind",
        "difficulty_band": "growth",
        "template_id": "zero_preserving_product_vs_category_baseline",
        "instruction": (
            "For every Northwind product, calculate 1997 order activity while preserving products with zero "
            "orders in that year. Apply the 1997 date condition when joining orders, and count or sum a line only "
            "when its order is in 1997. Net sales are unit price times quantity times one minus discount. Within "
            "each category, define the category baseline as the unweighted average of the product-level net sales, "
            "including zero-sale products. Return the ten products furthest above that baseline with columns "
            "`product_name`, `category`, `order_count`, `units_sold`, `net_sales`, `category_avg_net_sales`, and "
            "`sales_vs_category_avg`, rounding money to two decimals. Order by sales_vs_category_avg descending "
            "and product_name alphabetically."
        ),
        "sql": """WITH product_1997 AS (
    SELECT p.productid,
           p.productname,
           c.categoryname AS category,
           COUNT(DISTINCT o.orderid) AS order_count,
           COALESCE(SUM(CASE WHEN o.orderid IS NOT NULL THEN od.quantity ELSE 0 END), 0) AS units_sold,
           ROUND(COALESCE(SUM(CASE WHEN o.orderid IS NOT NULL THEN od.unitprice * od.quantity * (1 - od.discount) ELSE 0 END), 0), 2) AS net_sales
    FROM products AS p
    JOIN categories AS c ON c.categoryid = p.categoryid
    LEFT JOIN order_details AS od ON od.productid = p.productid
    LEFT JOIN orders AS o ON o.orderid = od.orderid
        AND o.orderdate >= '1997-01-01' AND o.orderdate < '1998-01-01'
    GROUP BY p.productid, p.productname, c.categoryname
), compared AS (
    SELECT productname,
           category,
           order_count,
           units_sold,
           net_sales,
           ROUND(AVG(net_sales) OVER (PARTITION BY category), 2) AS category_avg_net_sales
    FROM product_1997
)
SELECT productname AS product_name,
       category,
       order_count,
       units_sold,
       net_sales,
       category_avg_net_sales,
       ROUND(net_sales - category_avg_net_sales, 2) AS sales_vs_category_avg
FROM compared
ORDER BY sales_vs_category_avg DESC, product_name ASC
LIMIT 10""",
        "expected_columns": [
            "product_name",
            "category",
            "order_count",
            "units_sold",
            "net_sales",
            "category_avg_net_sales",
            "sales_vs_category_avg",
        ],
        "row_bounds": [10, 10],
        "operators": [
            "multi_stage_cte",
            "zero_preserving_left_join",
            "conditional_aggregate",
            "partitioned_window",
            "comparison_delta",
        ],
        "roles": [
            {
                "role": "product_population",
                "kind": "table",
                "query": "product category price stock",
                "selected": "products",
            },
            {
                "role": "category_dimension",
                "kind": "table",
                "query": "product category name",
                "selected": "categories",
            },
            {
                "role": "order_line_fact",
                "kind": "table",
                "query": "order product quantity unit price discount",
                "selected": "order_details",
            },
            {
                "role": "order_date_fact",
                "kind": "table",
                "query": "order date customer",
                "selected": "orders",
            },
            {
                "role": "discount_measure",
                "kind": "column",
                "table": "order_details",
                "query": "line discount",
                "selected": "discount",
                "type": "numeric",
            },
        ],
        "joins": [
            ["products", "categoryid", "categories", "categoryid"],
            ["products", "productid", "order_details", "productid"],
            ["order_details", "orderid", "orders", "orderid"],
        ],
        "mutations": [
            mutation(
                "wrong_sales_year",
                "filter",
                "o.orderdate >= '1997-01-01' AND o.orderdate < '1998-01-01'",
                "o.orderdate >= '1998-01-01' AND o.orderdate < '1999-01-01'",
            ),
            mutation(
                "ignore_line_discounts",
                "measure",
                "od.unitprice * od.quantity * (1 - od.discount)",
                "od.unitprice * od.quantity",
            ),
            mutation(
                "lowest_relative_sales",
                "sorting",
                "ORDER BY sales_vs_category_avg DESC",
                "ORDER BY sales_vs_category_avg ASC",
            ),
        ],
        "complexity_target": target("growth"),
        "semantic_risks": [
            "zero_count_population",
            "filter_in_join_not_where",
            "product_grain_before_category_baseline",
            "unweighted_baseline",
        ],
    },
]
