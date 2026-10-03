"""Curated pilot mappings; every choice is re-checked against the offline catalog and live DB."""

from __future__ import annotations

from typing import Any


def mutation(mutation_id: str, category: str, old: str, new: str) -> dict[str, str]:
    return {"id": mutation_id, "category": category, "old": old, "new": new}


PILOT_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "sample_id": "sqlite_pilot_001",
        "database_id": "chinook",
        "template_id": "multi_hop_grouped_leaderboard",
        "instruction": (
            "The music merchandising team needs a compact sales leaderboard across all invoices. Return the five "
            "genres with at least 50 units sold, ranked by invoiced sales. For each genre report the genre name, "
            "number of distinct tracks purchased, number of distinct purchasing customers, units sold, and "
            "invoiced sales (line unit price multiplied by quantity), rounded to two decimals. Break sales ties "
            "alphabetically by genre."
        ),
        "sql": """SELECT g.Name AS genre,
       COUNT(DISTINCT ii.TrackId) AS purchased_tracks,
       COUNT(DISTINCT i.CustomerId) AS purchasing_customers,
       SUM(ii.Quantity) AS units_sold,
       ROUND(SUM(ii.UnitPrice * ii.Quantity), 2) AS invoiced_sales
FROM invoice_items AS ii
JOIN tracks AS t ON t.TrackId = ii.TrackId
JOIN genres AS g ON g.GenreId = t.GenreId
JOIN invoices AS i ON i.InvoiceId = ii.InvoiceId
GROUP BY g.GenreId, g.Name
HAVING SUM(ii.Quantity) >= 50
ORDER BY invoiced_sales DESC, genre ASC
LIMIT 5""",
        "expected_columns": ["genre", "purchased_tracks", "purchasing_customers", "units_sold", "invoiced_sales"],
        "row_bounds": [5, 5],
        "operators": ["three_joins", "group_by", "distinct_count", "having", "top_k", "derived_measure"],
        "roles": [
            {"role": "line_fact", "kind": "table", "query": "invoice line unit price quantity track", "selected": "invoice_items"},
            {"role": "genre_dimension", "kind": "table", "query": "track genre name", "selected": "genres"},
            {"role": "sales_measure", "kind": "column", "table": "invoice_items", "query": "unit price", "selected": "UnitPrice", "type": "numeric"},
            {"role": "quantity_measure", "kind": "column", "table": "invoice_items", "query": "quantity units", "selected": "Quantity", "type": "numeric"},
        ],
        "joins": [
            ["invoice_items", "TrackId", "tracks", "TrackId"],
            ["tracks", "GenreId", "genres", "GenreId"],
            ["invoice_items", "InvoiceId", "invoices", "InvoiceId"],
        ],
        "mutations": [
            mutation("reverse_leaderboard", "sorting", "ORDER BY invoiced_sales DESC", "ORDER BY invoiced_sales ASC"),
            mutation("raise_minimum_units", "threshold", "HAVING SUM(ii.Quantity) >= 50", "HAVING SUM(ii.Quantity) >= 100"),
            mutation("average_instead_of_sales", "aggregation", "SUM(ii.UnitPrice * ii.Quantity)", "AVG(ii.UnitPrice * ii.Quantity)"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_002",
        "database_id": "northwind",
        "template_id": "period_category_margin_summary",
        "instruction": (
            "For orders placed in calendar year 1997, identify the five product categories with the highest net "
            "sales after line discounts. Report category, distinct order count, units sold, gross sales before "
            "discount, net sales after discount, and the discount amount as a percentage of gross sales. Round "
            "money and percentage outputs to two decimals, and use category name as the ascending tie-breaker."
        ),
        "sql": """SELECT c.categoryname AS category,
       COUNT(DISTINCT o.orderid) AS order_count,
       SUM(od.quantity) AS units_sold,
       ROUND(SUM(od.unitprice * od.quantity), 2) AS gross_sales,
       ROUND(SUM(od.unitprice * od.quantity * (1 - od.discount)), 2) AS net_sales,
       ROUND(100.0 * SUM(od.unitprice * od.quantity * od.discount) /
             NULLIF(SUM(od.unitprice * od.quantity), 0), 2) AS discount_share_pct
FROM orders AS o
JOIN order_details AS od ON od.orderid = o.orderid
JOIN products AS p ON p.productid = od.productid
JOIN categories AS c ON c.categoryid = p.categoryid
WHERE o.orderdate >= '1997-01-01' AND o.orderdate < '1998-01-01'
GROUP BY c.categoryid, c.categoryname
ORDER BY net_sales DESC, category ASC
LIMIT 5""",
        "expected_columns": ["category", "order_count", "units_sold", "gross_sales", "net_sales", "discount_share_pct"],
        "row_bounds": [5, 5],
        "operators": ["three_joins", "date_range", "group_by", "ratio", "top_k"],
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "orders order date customer", "selected": "orders"},
            {"role": "line_fact", "kind": "table", "query": "order product quantity unit price discount", "selected": "order_details"},
            {"role": "category_dimension", "kind": "table", "query": "product category name", "selected": "categories"},
            {"role": "discount_measure", "kind": "column", "table": "order_details", "query": "line discount", "selected": "discount", "type": "numeric"},
        ],
        "joins": [
            ["orders", "orderid", "order_details", "orderid"],
            ["order_details", "productid", "products", "productid"],
            ["products", "categoryid", "categories", "categoryid"],
        ],
        "mutations": [
            mutation("wrong_calendar_year", "filter", "o.orderdate >= '1997-01-01' AND o.orderdate < '1998-01-01'", "o.orderdate >= '1998-01-01' AND o.orderdate < '1999-01-01'"),
            mutation("ignore_discounts", "aggregation", "SUM(od.unitprice * od.quantity * (1 - od.discount))", "SUM(od.unitprice * od.quantity)"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY net_sales DESC", "ORDER BY net_sales ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_003",
        "database_id": "Brazilian_E_Commerce",
        "template_id": "state_fulfillment_and_revenue",
        "instruction": (
            "Summarize 2018 delivered marketplace orders by the state of the seller that supplied each item. Keep "
            "states associated with at least 100 distinct delivered orders and return the five with the most item "
            "merchandise revenue. Include seller state, distinct delivered orders, items sold, merchandise revenue "
            "from item price, freight value, and the percentage of distinct orders delivered on or before their "
            "estimated delivery date. Round monetary totals and the percentage to two decimals; break revenue ties "
            "by state code."
        ),
        "sql": """SELECT s.seller_state,
       COUNT(DISTINCT o.order_id) AS delivered_orders,
       COUNT(*) AS items_sold,
       ROUND(SUM(oi.price), 2) AS merchandise_revenue,
       ROUND(SUM(oi.freight_value), 2) AS freight_revenue,
       ROUND(100.0 * COUNT(DISTINCT CASE
           WHEN datetime(o.order_delivered_customer_date) <= datetime(o.order_estimated_delivery_date)
           THEN o.order_id END) / COUNT(DISTINCT o.order_id), 2) AS on_time_order_pct
FROM olist_orders AS o
JOIN olist_order_items AS oi ON oi.order_id = o.order_id
JOIN olist_sellers AS s ON s.seller_id = oi.seller_id
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp >= '2018-01-01'
  AND o.order_purchase_timestamp < '2019-01-01'
GROUP BY s.seller_state
HAVING COUNT(DISTINCT o.order_id) >= 100
ORDER BY merchandise_revenue DESC, s.seller_state ASC
LIMIT 5""",
        "expected_columns": ["seller_state", "delivered_orders", "items_sold", "merchandise_revenue", "freight_revenue", "on_time_order_pct"],
        "row_bounds": [5, 5],
        "operators": ["two_joins", "date_range", "conditional_distinct", "group_by", "having", "top_k"],
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "orders status purchase delivered estimated date", "selected": "olist_orders"},
            {"role": "item_fact", "kind": "table", "query": "order items price freight seller", "selected": "olist_order_items"},
            {"role": "seller_dimension", "kind": "table", "query": "seller state", "selected": "olist_sellers"},
            {"role": "revenue_measure", "kind": "column", "table": "olist_order_items", "query": "item price", "selected": "price", "type": "numeric"},
        ],
        "joins": [
            ["olist_orders", "order_id", "olist_order_items", "order_id"],
            ["olist_order_items", "seller_id", "olist_sellers", "seller_id"],
        ],
        "mutations": [
            mutation("wrong_purchase_year", "filter", "o.order_purchase_timestamp >= '2018-01-01'\n  AND o.order_purchase_timestamp < '2019-01-01'", "o.order_purchase_timestamp >= '2017-01-01'\n  AND o.order_purchase_timestamp < '2018-01-01'"),
            mutation("late_instead_of_on_time", "predicate", "<= datetime(o.order_estimated_delivery_date)", "> datetime(o.order_estimated_delivery_date)"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY merchandise_revenue DESC", "ORDER BY merchandise_revenue ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_004",
        "database_id": "f1",
        "template_id": "competition_period_leaderboard",
        "instruction": (
            "Compare Formula 1 constructors over the 2010 through 2020 seasons, inclusive. Among constructors "
            "that entered at least 20 distinct races, return the top five by total championship points. Show the "
            "constructor, distinct races entered, driver-result entries, championship points, podium finishes "
            "(classified positions 1–3), and wins (classified position 1). Round points to one decimal and break "
            "ties alphabetically by constructor."
        ),
        "sql": """SELECT c.name AS constructor,
       COUNT(DISTINCT r.race_id) AS races_entered,
       COUNT(*) AS driver_entries,
       ROUND(SUM(res.points), 1) AS championship_points,
       SUM(CASE WHEN res.position BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS podium_finishes,
       SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
FROM results AS res
JOIN races AS r ON r.race_id = res.race_id
JOIN constructors AS c ON c.constructor_id = res.constructor_id
WHERE r.year BETWEEN 2010 AND 2020
GROUP BY c.constructor_id, c.name
HAVING COUNT(DISTINCT r.race_id) >= 20
ORDER BY championship_points DESC, constructor ASC
LIMIT 5""",
        "expected_columns": ["constructor", "races_entered", "driver_entries", "championship_points", "podium_finishes", "wins"],
        "row_bounds": [5, 5],
        "operators": ["two_joins", "range_filter", "conditional_aggregate", "having", "top_k"],
        "roles": [
            {"role": "result_fact", "kind": "table", "query": "race driver constructor position points results", "selected": "results"},
            {"role": "race_dimension", "kind": "table", "query": "race season year", "selected": "races"},
            {"role": "constructor_dimension", "kind": "table", "query": "constructor team name", "selected": "constructors"},
            {"role": "points_measure", "kind": "column", "table": "results", "query": "championship points", "selected": "points", "type": "numeric"},
        ],
        "joins": [
            ["results", "race_id", "races", "race_id"],
            ["results", "constructor_id", "constructors", "constructor_id"],
        ],
        "mutations": [
            mutation("wrong_season_window", "filter", "r.year BETWEEN 2010 AND 2020", "r.year BETWEEN 2000 AND 2010"),
            mutation("seconds_as_wins", "predicate", "res.position = 1", "res.position = 2"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY championship_points DESC", "ORDER BY championship_points ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_005",
        "database_id": "Baseball",
        "template_id": "qualified_entity_rate_leaderboard",
        "instruction": (
            "Using regular-season batting from 2000 through 2010 inclusive, rank players who accumulated at least "
            "500 at-bats in that period. Return the ten highest batting averages, where batting average is total "
            "hits divided by total at-bats across the period. Include player name, first and last season represented, "
            "at-bats, hits, home runs, and batting average rounded to four decimals. Break ties by more at-bats and "
            "then player name."
        ),
        "sql": """SELECT p.name_first || ' ' || p.name_last AS player_name,
       MIN(b.year) AS first_season,
       MAX(b.year) AS last_season,
       CAST(SUM(b.ab) AS INTEGER) AS at_bats,
       CAST(SUM(b.h) AS INTEGER) AS hits,
       CAST(SUM(b.hr) AS INTEGER) AS home_runs,
       ROUND(1.0 * SUM(b.h) / NULLIF(SUM(b.ab), 0), 4) AS batting_average
FROM batting AS b
JOIN player AS p ON p.player_id = b.player_id
WHERE b.year BETWEEN 2000 AND 2010
GROUP BY b.player_id, p.name_first, p.name_last
HAVING SUM(b.ab) >= 500
ORDER BY batting_average DESC, at_bats DESC, player_name ASC
LIMIT 10""",
        "expected_columns": ["player_name", "first_season", "last_season", "at_bats", "hits", "home_runs", "batting_average"],
        "row_bounds": [10, 10],
        "operators": ["join", "range_filter", "group_by", "having", "ratio", "multi_key_top_k"],
        "roles": [
            {"role": "batting_fact", "kind": "table", "query": "player season at bats hits home runs batting", "selected": "batting"},
            {"role": "player_dimension", "kind": "table", "query": "player first last name", "selected": "player"},
            {"role": "denominator", "kind": "column", "table": "batting", "query": "at bats", "selected": "ab", "type": "numeric"},
            {"role": "numerator", "kind": "column", "table": "batting", "query": "hits", "selected": "h", "type": "numeric"},
        ],
        "joins": [["batting", "player_id", "player", "player_id"]],
        "mutations": [
            mutation("wrong_season_window", "filter", "b.year BETWEEN 2000 AND 2010", "b.year BETWEEN 1990 AND 2000"),
            mutation("overly_strict_qualification", "threshold", "HAVING SUM(b.ab) >= 500", "HAVING SUM(b.ab) >= 5000"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY batting_average DESC", "ORDER BY batting_average ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_006",
        "database_id": "Airlines",
        "template_id": "operational_delay_leaderboard",
        "instruction": (
            "For flights marked Arrived, compare destination airports that have at least 100 such flights. Return "
            "the five destination cities with the greatest average arrival delay, using the English city name. "
            "Delay is actual arrival minus scheduled arrival in minutes. Include arrived-flight count, average "
            "delay, percentage arriving after schedule, and worst delay; round the three delay/percentage metrics "
            "to two decimals and break ties by city. Treat the literal missing marker \\N as missing."
        ),
        "sql": """SELECT json_extract(a.city, '$.en') AS destination_city,
       COUNT(*) AS arrived_flights,
       ROUND(AVG((julianday(substr(f.actual_arrival, 1, 19)) - julianday(substr(f.scheduled_arrival, 1, 19))) * 1440.0), 2) AS avg_arrival_delay_minutes,
       ROUND(100.0 * AVG(CASE WHEN julianday(substr(f.actual_arrival, 1, 19)) > julianday(substr(f.scheduled_arrival, 1, 19)) THEN 1.0 ELSE 0.0 END), 2) AS late_arrival_pct,
       ROUND(MAX((julianday(substr(f.actual_arrival, 1, 19)) - julianday(substr(f.scheduled_arrival, 1, 19))) * 1440.0), 2) AS worst_delay_minutes
FROM flights AS f
JOIN airports_data AS a ON a.airport_code = f.arrival_airport
WHERE f.status = 'Arrived' AND f.actual_arrival <> '\\N'
GROUP BY a.airport_code, a.city
HAVING COUNT(*) >= 100
ORDER BY avg_arrival_delay_minutes DESC, destination_city ASC
LIMIT 5""",
        "expected_columns": ["destination_city", "arrived_flights", "avg_arrival_delay_minutes", "late_arrival_pct", "worst_delay_minutes"],
        "row_bounds": [5, 5],
        "operators": ["join", "json_extract", "timestamp_normalization", "group_by", "having", "top_k"],
        "roles": [
            {"role": "flight_fact", "kind": "table", "query": "flight scheduled actual arrival status destination airport", "selected": "flights"},
            {"role": "airport_dimension", "kind": "table", "query": "airport city code", "selected": "airports_data"},
            {"role": "actual_timestamp", "kind": "column", "table": "flights", "query": "actual arrival", "selected": "actual_arrival", "type": "temporal"},
            {"role": "scheduled_timestamp", "kind": "column", "table": "flights", "query": "scheduled arrival", "selected": "scheduled_arrival", "type": "temporal"},
        ],
        "joins": [["flights", "arrival_airport", "airports_data", "airport_code"]],
        "mutations": [
            mutation("scheduled_instead_of_arrived", "filter", "f.status = 'Arrived'", "f.status = 'Scheduled'"),
            mutation("departure_time_as_baseline", "timestamp", "f.scheduled_arrival", "f.scheduled_departure"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY avg_arrival_delay_minutes DESC", "ORDER BY avg_arrival_delay_minutes ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_007",
        "database_id": "California_Traffic_Collision",
        "template_id": "categorical_safety_rate_summary",
        "instruction": (
            "For collisions during 2020 and 2021 that occurred outside private property, compare lighting conditions "
            "represented by at least 100 collisions. Report lighting condition, collision count, fatal-collision "
            "count, severe-injury-collision count, average injured victims per collision, and the percentage of "
            "collisions that were fatal or had at least one severe injury. Count a collision once in that combined "
            "percentage even if both conditions hold. Round the average and percentage to two decimals, ranking "
            "highest combined severity first, then by collision count and lighting name."
        ),
        "sql": """SELECT lighting,
       COUNT(*) AS collision_count,
       SUM(CASE WHEN killed_victims > 0 THEN 1 ELSE 0 END) AS fatal_collisions,
       SUM(CASE WHEN severe_injury_count > 0 THEN 1 ELSE 0 END) AS severe_injury_collisions,
       ROUND(AVG(COALESCE(injured_victims, 0)), 2) AS avg_injured_victims,
       ROUND(100.0 * SUM(CASE WHEN killed_victims > 0 OR severe_injury_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS fatal_or_severe_pct
FROM collisions
WHERE collision_date >= '2020-01-01' AND collision_date < '2022-01-01'
  AND not_private_property = 1
  AND lighting IS NOT NULL
GROUP BY lighting
HAVING COUNT(*) >= 100
ORDER BY fatal_or_severe_pct DESC, collision_count DESC, lighting ASC""",
        "expected_columns": ["lighting", "collision_count", "fatal_collisions", "severe_injury_collisions", "avg_injured_victims", "fatal_or_severe_pct"],
        "row_bounds": [3, 12],
        "operators": ["single_table", "date_range", "conditional_aggregate", "group_by", "having", "rate"],
        "roles": [
            {"role": "collision_fact", "kind": "table", "query": "traffic collision lighting injury fatal date private property", "selected": "collisions"},
            {"role": "lighting_dimension", "kind": "column", "table": "collisions", "query": "lighting condition", "selected": "lighting", "type": "text"},
            {"role": "fatal_measure", "kind": "column", "table": "collisions", "query": "killed victims", "selected": "killed_victims", "type": "numeric"},
        ],
        "joins": [],
        "mutations": [
            mutation("wrong_year_window", "filter", "collision_date >= '2020-01-01' AND collision_date < '2022-01-01'", "collision_date >= '2018-01-01' AND collision_date < '2020-01-01'"),
            mutation("raise_minimum_collisions", "threshold", "HAVING COUNT(*) >= 100", "HAVING COUNT(*) >= 500"),
            mutation("fatal_only_rate", "predicate", "killed_victims > 0 OR severe_injury_count > 0", "killed_victims > 0"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_008",
        "database_id": "Pagila",
        "template_id": "multi_hop_category_utilization",
        "instruction": (
            "Create a top-five category report for completed rentals (rentals with a return timestamp). Rank film "
            "categories by realized payment revenue. For each category include distinct rented titles, distinct "
            "rentals, distinct renting customers, total payment revenue, and average elapsed hours from rental to "
            "return. Round revenue and average hours to two decimals and break revenue ties by category name."
        ),
        "sql": """SELECT c.name AS category,
       COUNT(DISTINCT f.film_id) AS rented_titles,
       COUNT(DISTINCT r.rental_id) AS rentals,
       COUNT(DISTINCT r.customer_id) AS renting_customers,
       ROUND(SUM(p.amount), 2) AS rental_revenue,
       ROUND(AVG((julianday(r.return_date) - julianday(r.rental_date)) * 24.0), 2) AS avg_rental_hours
FROM category AS c
JOIN film_category AS fc ON fc.category_id = c.category_id
JOIN film AS f ON f.film_id = fc.film_id
JOIN inventory AS inv ON inv.film_id = f.film_id
JOIN rental AS r ON r.inventory_id = inv.inventory_id
JOIN payment AS p ON p.rental_id = r.rental_id
WHERE r.return_date IS NOT NULL
GROUP BY c.category_id, c.name
ORDER BY rental_revenue DESC, category ASC
LIMIT 5""",
        "expected_columns": ["category", "rented_titles", "rentals", "renting_customers", "rental_revenue", "avg_rental_hours"],
        "row_bounds": [5, 5],
        "operators": ["five_joins", "not_null_filter", "distinct_count", "duration", "group_by", "top_k"],
        "roles": [
            {"role": "rental_fact", "kind": "table", "query": "rental customer inventory return date", "selected": "rental"},
            {"role": "payment_fact", "kind": "table", "query": "rental payment amount", "selected": "payment"},
            {"role": "category_dimension", "kind": "table", "query": "film category name", "selected": "category"},
            {"role": "revenue_measure", "kind": "column", "table": "payment", "query": "payment amount", "selected": "amount", "type": "numeric"},
        ],
        "joins": [
            ["category", "category_id", "film_category", "category_id"],
            ["film_category", "film_id", "film", "film_id"],
            ["film", "film_id", "inventory", "film_id"],
            ["inventory", "inventory_id", "rental", "inventory_id"],
            ["rental", "rental_id", "payment", "rental_id"],
        ],
        "mutations": [
            mutation("unreturned_rentals", "filter", "r.return_date IS NOT NULL", "r.return_date IS NULL"),
            mutation("average_payment_not_revenue", "aggregation", "SUM(p.amount)", "AVG(p.amount)"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY rental_revenue DESC", "ORDER BY rental_revenue ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_009",
        "database_id": "education_business",
        "template_id": "fact_dimension_unit_economics",
        "instruction": (
            "For fiscal year 2021 hardware sales, summarize every product category with positive unit sales. "
            "Report category, distinct products sold, units sold, gross sales value (quantity times the product's "
            "fiscal-year gross price), manufacturing cost (quantity times fiscal-year unit manufacturing cost), "
            "and a manufacturing gross-margin percentage computed from those two values. Round monetary values and "
            "margin to two decimals, sorting by gross sales value descending and category ascending. This is a "
            "manufacturing-margin proxy and should not apply customer discounts."
        ),
        "sql": """SELECT p.category,
       COUNT(DISTINCT s.product_code) AS products_sold,
       SUM(s.sold_quantity) AS units_sold,
       ROUND(SUM(s.sold_quantity * gp.gross_price), 2) AS gross_sales_value,
       ROUND(SUM(s.sold_quantity * mc.manufacturing_cost), 2) AS manufacturing_cost,
       ROUND(100.0 * SUM(s.sold_quantity * (gp.gross_price - mc.manufacturing_cost)) /
             NULLIF(SUM(s.sold_quantity * gp.gross_price), 0), 2) AS gross_margin_pct
FROM hardware_fact_sales_monthly AS s
JOIN hardware_dim_product AS p ON p.product_code = s.product_code
JOIN hardware_fact_gross_price AS gp
  ON gp.product_code = s.product_code AND gp.fiscal_year = s.fiscal_year
JOIN hardware_fact_manufacturing_cost AS mc
  ON mc.product_code = s.product_code AND mc.cost_year = s.fiscal_year
WHERE s.fiscal_year = 2021
GROUP BY p.category
HAVING SUM(s.sold_quantity) > 0
ORDER BY gross_sales_value DESC, p.category ASC""",
        "expected_columns": ["category", "products_sold", "units_sold", "gross_sales_value", "manufacturing_cost", "gross_margin_pct"],
        "row_bounds": [10, 20],
        "operators": ["three_joins", "multi_key_join", "period_filter", "derived_measure", "ratio", "group_by"],
        "roles": [
            {"role": "sales_fact", "kind": "table", "query": "hardware monthly product customer sold quantity fiscal year", "selected": "hardware_fact_sales_monthly"},
            {"role": "product_dimension", "kind": "table", "query": "hardware product category", "selected": "hardware_dim_product"},
            {"role": "price_fact", "kind": "table", "query": "product fiscal year gross price", "selected": "hardware_fact_gross_price"},
            {"role": "cost_fact", "kind": "table", "query": "product year manufacturing cost", "selected": "hardware_fact_manufacturing_cost"},
        ],
        "joins": [
            ["hardware_fact_sales_monthly", "product_code", "hardware_dim_product", "product_code"],
            ["hardware_fact_sales_monthly", "product_code", "hardware_fact_gross_price", "product_code"],
            ["hardware_fact_sales_monthly", "product_code", "hardware_fact_manufacturing_cost", "product_code"],
        ],
        "mutations": [
            mutation("wrong_fiscal_year", "filter", "s.fiscal_year = 2021", "s.fiscal_year = 2020"),
            mutation("cost_used_as_sales", "measure", "s.sold_quantity * gp.gross_price", "s.sold_quantity * mc.manufacturing_cost"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY gross_sales_value DESC", "ORDER BY gross_sales_value ASC"),
        ],
    },
    {
        "sample_id": "sqlite_pilot_010",
        "database_id": "delivery_center",
        "template_id": "operational_hub_scorecard",
        "instruction": (
            "Build a 2021 scorecard for hubs handling finished orders. Keep hubs with at least 1,000 distinct "
            "finished orders and return the five with the greatest order value. Include hub name, city, state, "
            "finished-order count, summed order amount, average order cycle time in minutes, average delivery "
            "distance in kilometers, and the percentage of associated delivery records marked DELIVERED. Round "
            "value, both averages, and percentage to two decimals; break order-value ties by hub name."
        ),
        "sql": """SELECT h.hub_name,
       h.hub_city,
       h.hub_state,
       COUNT(DISTINCT o.order_id) AS finished_orders,
       ROUND(SUM(o.order_amount), 2) AS order_value,
       ROUND(AVG(o.order_metric_cycle_time), 2) AS avg_cycle_minutes,
       ROUND(AVG(d.delivery_distance_meters) / 1000.0, 2) AS avg_delivery_km,
       ROUND(100.0 * AVG(CASE WHEN d.delivery_status = 'DELIVERED' THEN 1.0 ELSE 0.0 END), 2) AS delivered_pct
FROM orders AS o
JOIN stores AS s ON s.store_id = o.store_id
JOIN hubs AS h ON h.hub_id = s.hub_id
JOIN deliveries AS d ON d.delivery_order_id = o.delivery_order_id
WHERE o.order_status = 'FINISHED'
  AND o.order_created_year = 2021
GROUP BY h.hub_id, h.hub_name, h.hub_city, h.hub_state
HAVING COUNT(DISTINCT o.order_id) >= 1000
ORDER BY order_value DESC, h.hub_name ASC
LIMIT 5""",
        "expected_columns": ["hub_name", "hub_city", "hub_state", "finished_orders", "order_value", "avg_cycle_minutes", "avg_delivery_km", "delivered_pct"],
        "row_bounds": [5, 5],
        "operators": ["three_joins", "period_filter", "group_by", "having", "conditional_rate", "unit_conversion", "top_k"],
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "finished order amount year cycle delivery store", "selected": "orders"},
            {"role": "delivery_fact", "kind": "table", "query": "delivery status distance order", "selected": "deliveries"},
            {"role": "store_dimension", "kind": "table", "query": "store hub", "selected": "stores"},
            {"role": "hub_dimension", "kind": "table", "query": "hub name city state", "selected": "hubs"},
        ],
        "joins": [
            ["orders", "store_id", "stores", "store_id"],
            ["stores", "hub_id", "hubs", "hub_id"],
            ["orders", "delivery_order_id", "deliveries", "delivery_order_id"],
        ],
        "mutations": [
            mutation("wrong_order_year", "filter", "o.order_created_year = 2021", "o.order_created_year = 2020"),
            mutation("failed_delivery_rate", "predicate", "d.delivery_status = 'DELIVERED'", "d.delivery_status = 'FAILED'"),
            mutation("reverse_leaderboard", "sorting", "ORDER BY order_value DESC", "ORDER BY order_value ASC"),
        ],
    },
]
