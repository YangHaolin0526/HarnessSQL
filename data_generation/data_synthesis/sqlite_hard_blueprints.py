"""Second SQLite pilot: manually authored to match Spider 2.0 complexity axes."""

from __future__ import annotations

from typing import Any

from .sqlite_pilot_blueprints import mutation


HARD_TARGET = {
    "reference": "106 officially result-verified Spider 2.0 SQLite predictions",
    "minimum_structural_score": 22.833,
    "minimum_sql_tokens": 231.5,
    "minimum_advanced_families": 2,
}


HARD_PILOT_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "sample_id": "sqlite_hard_001",
        "database_id": "chinook",
        "template_id": "customer_lifetime_halfway_milestone",
        "instruction": (
            "Using invoiced line-item sales, aggregate each customer's purchases by calendar month and calculate "
            "their cumulative spend in chronological order. Keep customers with at least five distinct active "
            "months, and identify the first active month in which cumulative spend reaches at least 50% of that "
            "customer's lifetime sales. Return the ten customers who took the most whole calendar months to reach "
            "that point. Report customer name, country, active-month count, first sales month, halfway month, "
            "calendar months elapsed, lifetime sales, cumulative sales at the crossing, and the cumulative "
            "percentage. Round monetary and percentage values to two decimals. Break ties by higher lifetime "
            "sales, then customer name alphabetically."
        ),
        "sql": """WITH monthly_spend AS (
    SELECT i.CustomerId AS customer_id,
           c.FirstName || ' ' || c.LastName AS customer_name,
           c.Country AS country,
           date(i.InvoiceDate, 'start of month') AS sales_month,
           SUM(ii.UnitPrice * ii.Quantity) AS monthly_sales
    FROM invoices AS i
    JOIN invoice_items AS ii ON ii.InvoiceId = i.InvoiceId
    JOIN customers AS c ON c.CustomerId = i.CustomerId
    GROUP BY i.CustomerId, c.FirstName, c.LastName, c.Country,
             date(i.InvoiceDate, 'start of month')
),
customer_totals AS (
    SELECT customer_id,
           SUM(monthly_sales) AS lifetime_sales,
           COUNT(*) AS active_months,
           MIN(sales_month) AS first_sales_month
    FROM monthly_spend
    GROUP BY customer_id
),
spend_progress AS (
    SELECT m.customer_id, m.customer_name, m.country, m.sales_month,
           m.monthly_sales, t.lifetime_sales, t.active_months, t.first_sales_month,
           SUM(m.monthly_sales) OVER (
               PARTITION BY m.customer_id ORDER BY m.sales_month
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS cumulative_sales
    FROM monthly_spend AS m
    JOIN customer_totals AS t ON t.customer_id = m.customer_id
),
halfway_candidates AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY customer_id ORDER BY sales_month
           ) AS crossing_rank
    FROM spend_progress
    WHERE cumulative_sales >= lifetime_sales * 0.5
),
first_crossing AS (
    SELECT * FROM halfway_candidates WHERE crossing_rank = 1
)
SELECT customer_name,
       country,
       active_months,
       first_sales_month,
       sales_month AS half_sales_month,
       (CAST(strftime('%Y', sales_month) AS INTEGER) -
        CAST(strftime('%Y', first_sales_month) AS INTEGER)) * 12 +
       CAST(strftime('%m', sales_month) AS INTEGER) -
       CAST(strftime('%m', first_sales_month) AS INTEGER) AS months_to_half,
       ROUND(lifetime_sales, 2) AS lifetime_sales,
       ROUND(cumulative_sales, 2) AS sales_at_crossing,
       ROUND(100.0 * cumulative_sales / NULLIF(lifetime_sales, 0), 2) AS cumulative_pct
FROM first_crossing
WHERE active_months >= 5
ORDER BY months_to_half DESC, lifetime_sales DESC, customer_name ASC
LIMIT 10""",
        "expected_columns": ["customer_name", "country", "active_months", "first_sales_month", "half_sales_month", "months_to_half", "lifetime_sales", "sales_at_crossing", "cumulative_pct"],
        "row_bounds": [10, 10],
        "operators": ["multi_stage_cte", "monthly_grain", "cumulative_window", "first_threshold_crossing", "calendar_month_difference", "multi_key_top_k"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "grain_safe_multi_fact"],
        "semantic_risks": ["metric_source", "entity_identity", "date_boundary", "window_cumulative", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "invoice_fact", "kind": "table", "query": "invoices customer invoice date total", "selected": "invoices"},
            {"role": "line_fact", "kind": "table", "query": "invoice items unit price quantity", "selected": "invoice_items"},
            {"role": "customer_dimension", "kind": "table", "query": "customers first last name country", "selected": "customers"},
            {"role": "sales_value", "kind": "column", "table": "invoice_items", "type": "numeric", "query": "unit price", "selected": "UnitPrice"},
        ],
        "joins": [
            ["invoices", "InvoiceId", "invoice_items", "InvoiceId"],
            ["invoices", "CustomerId", "customers", "CustomerId"],
        ],
        "mutations": [
            mutation("later_crossing", "threshold", "lifetime_sales * 0.5", "lifetime_sales * 0.75"),
            mutation("impossible_activity_minimum", "population", "active_months >= 5", "active_months >= 8"),
            mutation("shortest_delay_first", "ranking", "ORDER BY months_to_half DESC", "ORDER BY months_to_half ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_002",
        "database_id": "northwind",
        "template_id": "dense_calendar_category_sales_spike",
        "instruction": (
            "For every product category, construct all twelve calendar months of 1997, treating a month with no "
            "orders as zero net sales. Net sales are line unit price times quantity after the line discount. For "
            "each month from April onward, compare net sales with the average of the immediately preceding three "
            "calendar months; exclude comparisons whose prior-three-month average is zero. Select each category's "
            "largest percentage spike, breaking an internal tie by the earlier month, then return the five category "
            "spikes with the largest percentages. Report category, spike month, distinct order count in that month, "
            "net sales, prior-three-month average, absolute spike, and percentage spike. Round numeric sales and "
            "percentage outputs to two decimals; break final ties by category name."
        ),
        "sql": """WITH RECURSIVE months(month_start) AS (
    SELECT date('1997-01-01')
    UNION ALL
    SELECT date(month_start, '+1 month')
    FROM months
    WHERE month_start < '1997-12-01'
),
category_month_sales AS (
    SELECT p.categoryid,
           date(o.orderdate, 'start of month') AS month_start,
           SUM(od.unitprice * od.quantity * (1.0 - od.discount)) AS net_sales,
           COUNT(DISTINCT o.orderid) AS order_count
    FROM orders AS o
    JOIN order_details AS od ON od.orderid = o.orderid
    JOIN products AS p ON p.productid = od.productid
    WHERE o.orderdate >= '1997-01-01' AND o.orderdate < '1998-01-01'
    GROUP BY p.categoryid, date(o.orderdate, 'start of month')
),
dense_category_months AS (
    SELECT c.categoryid,
           c.categoryname,
           m.month_start,
           COALESCE(s.net_sales, 0.0) AS net_sales,
           COALESCE(s.order_count, 0) AS order_count
    FROM categories AS c
    JOIN months AS m ON 1 = 1
    LEFT JOIN category_month_sales AS s
      ON s.categoryid = c.categoryid AND s.month_start = m.month_start
),
rolling_baseline AS (
    SELECT *,
           AVG(net_sales) OVER (
               PARTITION BY categoryid ORDER BY month_start
               ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
           ) AS prior_three_month_avg,
           COUNT(*) OVER (
               PARTITION BY categoryid ORDER BY month_start
               ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
           ) AS prior_month_count
    FROM dense_category_months
),
qualified_spikes AS (
    SELECT *,
           net_sales - prior_three_month_avg AS spike_amount,
           100.0 * (net_sales - prior_three_month_avg) /
               NULLIF(prior_three_month_avg, 0) AS spike_pct
    FROM rolling_baseline
    WHERE prior_month_count = 3 AND prior_three_month_avg > 0
),
ranked_spikes AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY categoryid
               ORDER BY spike_pct DESC, month_start ASC
           ) AS category_spike_rank
    FROM qualified_spikes
)
SELECT categoryname AS category,
       month_start AS spike_month,
       order_count,
       ROUND(net_sales, 2) AS net_sales,
       ROUND(prior_three_month_avg, 2) AS prior_three_month_avg,
       ROUND(spike_amount, 2) AS spike_amount,
       ROUND(spike_pct, 2) AS spike_pct
FROM ranked_spikes
WHERE category_spike_rank = 1
ORDER BY spike_pct DESC, category ASC
LIMIT 5""",
        "expected_columns": ["category", "spike_month", "order_count", "net_sales", "prior_three_month_avg", "spike_amount", "spike_pct"],
        "row_bounds": [5, 5],
        "operators": ["recursive_calendar", "dense_dimension_grid", "discounted_sales", "rolling_prior_window", "per_group_argmax", "top_k"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "recursive", "grain_safe_multi_fact"],
        "semantic_risks": ["date_boundary", "metric_source", "window_cumulative", "denominator", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "orders order date", "selected": "orders"},
            {"role": "line_fact", "kind": "table", "query": "order details quantity unitprice discount", "selected": "order_details"},
            {"role": "product_dimension", "kind": "table", "query": "products category product id", "selected": "products"},
            {"role": "category_dimension", "kind": "table", "query": "categories category name", "selected": "categories"},
            {"role": "discount", "kind": "column", "table": "order_details", "type": "numeric", "query": "discount", "selected": "discount"},
        ],
        "joins": [
            ["orders", "orderid", "order_details", "orderid"],
            ["order_details", "productid", "products", "productid"],
            ["products", "categoryid", "categories", "categoryid"],
        ],
        "mutations": [
            mutation("ignore_discounts", "metric_source", "od.unitprice * od.quantity * (1.0 - od.discount)", "od.unitprice * od.quantity"),
            mutation("two_month_baseline", "window", "ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING", "ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING"),
            mutation("smallest_spike", "ranking", "ORDER BY spike_pct DESC", "ORDER BY spike_pct ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_003",
        "database_id": "Brazilian_E_Commerce",
        "template_id": "seller_quarter_same_period_retention",
        "instruction": (
            "Build quarterly cohorts of sellers whose first delivered order occurred in the first or second quarter "
            "of 2017. A seller is retained only if they also had at least one delivered order in the same calendar "
            "quarter of 2018. Group sellers by their state and first-sale quarter, keep groups with at least five "
            "cohort sellers, and within each cohort quarter return the three states with the highest seller retention "
            "rate, breaking ties by larger cohort and then state code. Report cohort quarter, seller state, cohort "
            "seller count, retained seller count, initial-quarter distinct delivered orders, initial-quarter item-price "
            "revenue, same-quarter-next-year item-price revenue, retention percentage, and the unweighted average "
            "seller-level revenue growth percentage among retained sellers. Round revenue and percentages to two "
            "decimals and order by cohort quarter and the stated rank."
        ),
        "sql": """WITH delivered_seller_quarters AS (
    SELECT oi.seller_id,
           s.seller_state,
           printf('%04d-%02d-01',
                  CAST(strftime('%Y', o.order_purchase_timestamp) AS INTEGER),
                  ((CAST(strftime('%m', o.order_purchase_timestamp) AS INTEGER) - 1) / 3) * 3 + 1
           ) AS quarter_start,
           COUNT(DISTINCT o.order_id) AS delivered_orders,
           SUM(oi.price) AS merchandise_revenue
    FROM olist_orders AS o
    JOIN olist_order_items AS oi ON oi.order_id = o.order_id
    JOIN olist_sellers AS s ON s.seller_id = oi.seller_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY oi.seller_id, s.seller_state, quarter_start
),
seller_first_quarter AS (
    SELECT seller_id, MIN(quarter_start) AS first_quarter
    FROM delivered_seller_quarters
    GROUP BY seller_id
),
cohort_performance AS (
    SELECT initial.seller_id,
           initial.seller_state,
           initial.quarter_start AS cohort_quarter,
           initial.delivered_orders AS cohort_orders,
           initial.merchandise_revenue AS cohort_revenue,
           followup.delivered_orders AS next_year_orders,
           followup.merchandise_revenue AS next_year_revenue
    FROM delivered_seller_quarters AS initial
    JOIN seller_first_quarter AS first
      ON first.seller_id = initial.seller_id
     AND first.first_quarter = initial.quarter_start
    LEFT JOIN delivered_seller_quarters AS followup
      ON followup.seller_id = initial.seller_id
     AND followup.quarter_start = date(initial.quarter_start, '+1 year')
    WHERE initial.quarter_start >= '2017-01-01'
      AND initial.quarter_start < '2017-07-01'
),
state_cohorts AS (
    SELECT seller_state,
           cohort_quarter,
           COUNT(*) AS cohort_sellers,
           SUM(CASE WHEN next_year_orders IS NOT NULL THEN 1 ELSE 0 END) AS retained_sellers,
           SUM(cohort_orders) AS cohort_orders,
           ROUND(SUM(cohort_revenue), 2) AS cohort_revenue,
           ROUND(SUM(COALESCE(next_year_revenue, 0)), 2) AS next_year_revenue,
           100.0 * SUM(CASE WHEN next_year_orders IS NOT NULL THEN 1 ELSE 0 END) /
               COUNT(*) AS retention_pct,
           AVG(CASE WHEN next_year_revenue IS NOT NULL AND cohort_revenue > 0
                    THEN 100.0 * (next_year_revenue - cohort_revenue) / cohort_revenue END
           ) AS retained_seller_avg_growth_pct
    FROM cohort_performance
    GROUP BY seller_state, cohort_quarter
    HAVING COUNT(*) >= 5
),
ranked_state_cohorts AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY cohort_quarter
               ORDER BY retention_pct DESC, cohort_sellers DESC, seller_state ASC
           ) AS retention_rank
    FROM state_cohorts
)
SELECT cohort_quarter,
       seller_state,
       cohort_sellers,
       retained_sellers,
       cohort_orders,
       cohort_revenue,
       next_year_revenue,
       ROUND(retention_pct, 2) AS retention_pct,
       ROUND(retained_seller_avg_growth_pct, 2) AS retained_seller_avg_growth_pct
FROM ranked_state_cohorts
WHERE retention_rank <= 3
ORDER BY cohort_quarter ASC, retention_rank ASC, seller_state ASC""",
        "expected_columns": ["cohort_quarter", "seller_state", "cohort_sellers", "retained_sellers", "cohort_orders", "cohort_revenue", "next_year_revenue", "retention_pct", "retained_seller_avg_growth_pct"],
        "row_bounds": [6, 6],
        "operators": ["quarter_derivation", "first_event_cohort", "same_period_retention", "seller_level_growth", "conditional_aggregation", "partitioned_top_k"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "grain_safe_multi_fact", "nontrivial_derived_classification"],
        "semantic_risks": ["population", "entity_identity", "distinct_dedup", "denominator", "date_boundary", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "olist orders status purchase timestamp delivered", "selected": "olist_orders"},
            {"role": "item_fact", "kind": "table", "query": "olist order items seller price", "selected": "olist_order_items"},
            {"role": "seller_dimension", "kind": "table", "query": "olist sellers state seller id", "selected": "olist_sellers"},
            {"role": "revenue", "kind": "column", "table": "olist_order_items", "type": "numeric", "query": "price", "selected": "price"},
        ],
        "joins": [
            ["olist_orders", "order_id", "olist_order_items", "order_id"],
            ["olist_order_items", "seller_id", "olist_sellers", "seller_id"],
        ],
        "mutations": [
            mutation("next_quarter_not_next_year", "retention_window", "date(initial.quarter_start, '+1 year')", "date(initial.quarter_start, '+3 months')"),
            mutation("include_third_quarter", "cohort_boundary", "initial.quarter_start < '2017-07-01'", "initial.quarter_start < '2017-10-01'"),
            mutation("lowest_retention_states", "ranking", "ORDER BY retention_pct DESC", "ORDER BY retention_pct ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_004",
        "database_id": "f1",
        "template_id": "driver_consecutive_points_improvement_streak",
        "instruction": (
            "Aggregate Formula 1 results by driver and season, retaining driver-seasons with at least eight distinct "
            "race starts. An improvement is a season whose total points are strictly greater than the immediately "
            "preceding calendar season for that driver; a missing year or non-increase breaks the streak. Find each "
            "driver's best streak containing at least two consecutive year-over-year improvements, preferring more "
            "improvements, then larger total points gain, then the later ending season. Return the ten best driver "
            "streaks overall by those same first two criteria and driver name. Report driver name, start and end "
            "season, number of consecutive improvements, points in the season before the first increase, points in "
            "the final season, and total gain across the streak, with point values rounded to one decimal."
        ),
        "sql": """WITH driver_seasons AS (
    SELECT res.driver_id,
           d.forename || ' ' || d.surname AS driver_name,
           r.year AS season,
           COUNT(DISTINCT r.race_id) AS starts,
           SUM(CASE WHEN res.position IS NOT NULL THEN 1 ELSE 0 END) AS classified_finishes,
           SUM(res.points) AS season_points
    FROM results AS res
    JOIN races AS r ON r.race_id = res.race_id
    JOIN drivers AS d ON d.driver_id = res.driver_id
    GROUP BY res.driver_id, d.forename, d.surname, r.year
    HAVING COUNT(DISTINCT r.race_id) >= 8
),
season_comparisons AS (
    SELECT *,
           LAG(season) OVER (PARTITION BY driver_id ORDER BY season) AS previous_season,
           LAG(season_points) OVER (PARTITION BY driver_id ORDER BY season) AS previous_points
    FROM driver_seasons
),
improvement_flags AS (
    SELECT *,
           CASE WHEN previous_season = season - 1
                     AND season_points > previous_points
                THEN 1 ELSE 0 END AS is_improvement
    FROM season_comparisons
),
streak_labels AS (
    SELECT *,
           SUM(CASE WHEN is_improvement = 0 THEN 1 ELSE 0 END) OVER (
               PARTITION BY driver_id ORDER BY season
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS streak_group
    FROM improvement_flags
),
improvement_streaks AS (
    SELECT driver_id,
           driver_name,
           streak_group,
           MIN(CASE WHEN is_improvement = 1 THEN previous_season END) AS streak_start_season,
           MAX(CASE WHEN is_improvement = 1 THEN season END) AS streak_end_season,
           SUM(is_improvement) AS consecutive_improvements,
           SUM(CASE WHEN is_improvement = 1 THEN season_points - previous_points ELSE 0 END) AS total_points_gain,
           MIN(CASE WHEN is_improvement = 1 THEN previous_points END) AS starting_points,
           MAX(CASE WHEN is_improvement = 1 THEN season_points END) AS ending_points
    FROM streak_labels
    GROUP BY driver_id, driver_name, streak_group
    HAVING SUM(is_improvement) >= 2
),
ranked_driver_streaks AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY driver_id
               ORDER BY consecutive_improvements DESC,
                        total_points_gain DESC,
                        streak_end_season DESC
           ) AS driver_streak_rank
    FROM improvement_streaks
),
best_driver_streaks AS (
    SELECT * FROM ranked_driver_streaks WHERE driver_streak_rank = 1
)
SELECT driver_name,
       streak_start_season,
       streak_end_season,
       consecutive_improvements,
       ROUND(starting_points, 1) AS starting_points,
       ROUND(ending_points, 1) AS ending_points,
       ROUND(total_points_gain, 1) AS total_points_gain
FROM best_driver_streaks
ORDER BY consecutive_improvements DESC,
         total_points_gain DESC,
         driver_name ASC
LIMIT 10""",
        "expected_columns": ["driver_name", "streak_start_season", "streak_end_season", "consecutive_improvements", "starting_points", "ending_points", "total_points_gain"],
        "row_bounds": [10, 10],
        "operators": ["season_grain", "lag", "gaps_and_islands", "conditional_aggregation", "per_entity_argmax", "global_top_k"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "nontrivial_derived_classification"],
        "semantic_risks": ["population", "metric_source", "date_boundary", "window_cumulative", "ranking_tie", "entity_identity"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "result_fact", "kind": "table", "query": "results race driver position points", "selected": "results"},
            {"role": "race_dimension", "kind": "table", "query": "races year season race id", "selected": "races"},
            {"role": "driver_dimension", "kind": "table", "query": "drivers forename surname driver id", "selected": "drivers"},
            {"role": "points", "kind": "column", "table": "results", "type": "numeric", "query": "points", "selected": "points"},
        ],
        "joins": [
            ["results", "race_id", "races", "race_id"],
            ["results", "driver_id", "drivers", "driver_id"],
        ],
        "mutations": [
            mutation("reverse_improvement", "comparison", "season_points > previous_points", "season_points < previous_points"),
            mutation("fifteen_start_seasons", "population", "HAVING COUNT(DISTINCT r.race_id) >= 8", "HAVING COUNT(DISTINCT r.race_id) >= 15"),
            mutation("shortest_streaks_first", "ranking", "ORDER BY consecutive_improvements DESC", "ORDER BY consecutive_improvements ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_005",
        "database_id": "education_business",
        "template_id": "regional_longest_account_activity_streak",
        "instruction": (
            "For each sales region, find the customer account with the best continuous monthly ordering streak in "
            "calendar year 2016. A month is active when the account placed at least one order, and consecutive means "
            "adjacent calendar months; gaps split streaks. Rank streaks within a region by more consecutive active "
            "months, then higher total sales during the streak, account name, and earlier start. Report one winner "
            "per region with region, account, streak start and end months, active-month count, order count, streak "
            "sales, average monthly streak sales, the region's unweighted average sales across all active "
            "account-month observations in 2016, and the winner's monthly average as a percentage of that baseline. "
            "Round money and percentage fields to two decimals and order regions alphabetically."
        ),
        "sql": """WITH account_months AS (
    SELECT a.id AS account_id,
           a.name AS account_name,
           r.id AS region_id,
           r.name AS region_name,
           date(o.occurred_at, 'start of month') AS order_month,
           COUNT(*) AS order_count,
           SUM(o.total_amt_usd) AS monthly_sales
    FROM web_orders AS o
    JOIN web_accounts AS a ON a.id = o.account_id
    JOIN web_sales_reps AS sr ON sr.id = a.sales_rep_id
    JOIN web_region AS r ON r.id = sr.region_id
    WHERE o.occurred_at >= '2016-01-01' AND o.occurred_at < '2017-01-01'
    GROUP BY a.id, a.name, r.id, r.name, date(o.occurred_at, 'start of month')
),
sequenced_months AS (
    SELECT *,
           LAG(order_month) OVER (
               PARTITION BY account_id ORDER BY order_month
           ) AS previous_active_month
    FROM account_months
),
island_markers AS (
    SELECT *,
           CASE WHEN previous_active_month IS NULL
                     OR order_month <> date(previous_active_month, '+1 month')
                THEN 1 ELSE 0 END AS new_streak
    FROM sequenced_months
),
labeled_months AS (
    SELECT *,
           SUM(new_streak) OVER (
               PARTITION BY account_id ORDER BY order_month
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS streak_id
    FROM island_markers
),
account_streaks AS (
    SELECT account_id,
           account_name,
           region_id,
           region_name,
           streak_id,
           MIN(order_month) AS streak_start_month,
           MAX(order_month) AS streak_end_month,
           COUNT(*) AS consecutive_active_months,
           SUM(order_count) AS streak_orders,
           SUM(monthly_sales) AS streak_sales,
           AVG(monthly_sales) AS avg_monthly_streak_sales
    FROM labeled_months
    GROUP BY account_id, account_name, region_id, region_name, streak_id
),
region_active_month_baseline AS (
    SELECT region_id,
           AVG(monthly_sales) AS region_avg_active_account_month_sales
    FROM account_months
    GROUP BY region_id
),
ranked_streaks AS (
    SELECT s.*,
           b.region_avg_active_account_month_sales,
           ROW_NUMBER() OVER (
               PARTITION BY s.region_id
               ORDER BY s.consecutive_active_months DESC,
                        s.streak_sales DESC,
                        s.account_name ASC,
                        s.streak_start_month ASC
           ) AS region_rank
    FROM account_streaks AS s
    JOIN region_active_month_baseline AS b ON b.region_id = s.region_id
)
SELECT region_name,
       account_name,
       streak_start_month,
       streak_end_month,
       consecutive_active_months,
       streak_orders,
       ROUND(streak_sales, 2) AS streak_sales,
       ROUND(avg_monthly_streak_sales, 2) AS avg_monthly_streak_sales,
       ROUND(region_avg_active_account_month_sales, 2) AS region_avg_active_account_month_sales,
       ROUND(100.0 * avg_monthly_streak_sales /
             NULLIF(region_avg_active_account_month_sales, 0), 2) AS pct_of_region_baseline
FROM ranked_streaks
WHERE region_rank = 1
ORDER BY region_name ASC""",
        "expected_columns": ["region_name", "account_name", "streak_start_month", "streak_end_month", "consecutive_active_months", "streak_orders", "streak_sales", "avg_monthly_streak_sales", "region_avg_active_account_month_sales", "pct_of_region_baseline"],
        "row_bounds": [4, 4],
        "operators": ["monthly_grain", "lag", "gaps_and_islands", "regional_baseline", "per_partition_argmax", "ratio"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "grain_safe_multi_fact"],
        "semantic_risks": ["population", "metric_source", "date_boundary", "window_cumulative", "denominator", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "web orders account occurred total amount usd", "selected": "web_orders"},
            {"role": "account_dimension", "kind": "table", "query": "web accounts name sales rep", "selected": "web_accounts"},
            {"role": "sales_rep_dimension", "kind": "table", "query": "web sales reps region id", "selected": "web_sales_reps"},
            {"role": "region_dimension", "kind": "table", "query": "web region name", "selected": "web_region"},
            {"role": "sales_amount", "kind": "column", "table": "web_orders", "type": "numeric", "query": "total amount usd", "selected": "total_amt_usd"},
        ],
        "joins": [
            ["web_orders", "account_id", "web_accounts", "id"],
            ["web_accounts", "sales_rep_id", "web_sales_reps", "id"],
            ["web_sales_reps", "region_id", "web_region", "id"],
        ],
        "mutations": [
            mutation("use_2015", "date_boundary", "o.occurred_at >= '2016-01-01' AND o.occurred_at < '2017-01-01'", "o.occurred_at >= '2015-01-01' AND o.occurred_at < '2016-01-01'"),
            mutation("two_month_gap_is_consecutive", "sequence", "date(previous_active_month, '+1 month')", "date(previous_active_month, '+2 months')"),
            mutation("shortest_regional_streak", "ranking", "ORDER BY s.consecutive_active_months DESC", "ORDER BY s.consecutive_active_months ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_006",
        "database_id": "delivery_center",
        "template_id": "hub_worst_monthly_cycle_deterioration",
        "instruction": (
            "Using finished orders from 2021, calculate monthly operating metrics for each hub-month with at least "
            "100 orders. Contribution before payment fees is order amount plus delivery fee minus delivery cost. "
            "For every hub, compare each month with its immediately preceding calendar month and select the month "
            "with the greatest increase in average order cycle time, breaking a hub-level tie by the earlier month. "
            "Return the ten hubs with the largest selected deterioration. Report hub, previous and current month "
            "numbers, current finished-order count, previous and current average cycle minutes, deterioration in "
            "minutes, percentage change in total contribution, and current average production and transit time as "
            "percentages of current average cycle time. Round all calculated metrics to two decimals and break final "
            "ties by hub name."
        ),
        "sql": """WITH finished_orders AS (
    SELECT h.hub_id,
           h.hub_name,
           o.order_created_month AS order_month,
           o.order_id,
           o.order_metric_cycle_time AS cycle_minutes,
           o.order_metric_production_time AS production_minutes,
           o.order_metric_transit_time AS transit_minutes,
           o.order_amount + o.order_delivery_fee - o.order_delivery_cost AS contribution_before_payment_fees
    FROM orders AS o
    JOIN stores AS s ON s.store_id = o.store_id
    JOIN hubs AS h ON h.hub_id = s.hub_id
    WHERE o.order_status = 'FINISHED'
      AND o.order_created_year = 2021
      AND o.order_metric_cycle_time > 0
),
hub_month_metrics AS (
    SELECT hub_id,
           hub_name,
           order_month,
           COUNT(*) AS finished_orders,
           AVG(cycle_minutes) AS avg_cycle_minutes,
           AVG(production_minutes) AS avg_production_minutes,
           AVG(transit_minutes) AS avg_transit_minutes,
           SUM(contribution_before_payment_fees) AS total_contribution
    FROM finished_orders
    GROUP BY hub_id, hub_name, order_month
    HAVING COUNT(*) >= 100
),
month_comparisons AS (
    SELECT *,
           LAG(order_month) OVER (PARTITION BY hub_id ORDER BY order_month) AS previous_month,
           LAG(avg_cycle_minutes) OVER (PARTITION BY hub_id ORDER BY order_month) AS previous_avg_cycle_minutes,
           LAG(total_contribution) OVER (PARTITION BY hub_id ORDER BY order_month) AS previous_contribution
    FROM hub_month_metrics
),
consecutive_changes AS (
    SELECT *,
           avg_cycle_minutes - previous_avg_cycle_minutes AS cycle_deterioration_minutes,
           100.0 * (total_contribution - previous_contribution) /
               NULLIF(previous_contribution, 0) AS contribution_change_pct,
           100.0 * avg_production_minutes / NULLIF(avg_cycle_minutes, 0) AS production_share_pct,
           100.0 * avg_transit_minutes / NULLIF(avg_cycle_minutes, 0) AS transit_share_pct
    FROM month_comparisons
    WHERE previous_month = order_month - 1
),
ranked_hub_changes AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY hub_id
               ORDER BY cycle_deterioration_minutes DESC, order_month ASC
           ) AS hub_change_rank
    FROM consecutive_changes
),
worst_change_per_hub AS (
    SELECT * FROM ranked_hub_changes WHERE hub_change_rank = 1
)
SELECT hub_name,
       previous_month,
       order_month,
       finished_orders,
       ROUND(previous_avg_cycle_minutes, 2) AS previous_avg_cycle_minutes,
       ROUND(avg_cycle_minutes, 2) AS avg_cycle_minutes,
       ROUND(cycle_deterioration_minutes, 2) AS cycle_deterioration_minutes,
       ROUND(contribution_change_pct, 2) AS contribution_change_pct,
       ROUND(production_share_pct, 2) AS production_share_pct,
       ROUND(transit_share_pct, 2) AS transit_share_pct
FROM worst_change_per_hub
ORDER BY cycle_deterioration_minutes DESC, hub_name ASC
LIMIT 10""",
        "expected_columns": ["hub_name", "previous_month", "order_month", "finished_orders", "previous_avg_cycle_minutes", "avg_cycle_minutes", "cycle_deterioration_minutes", "contribution_change_pct", "production_share_pct", "transit_share_pct"],
        "row_bounds": [10, 10],
        "operators": ["monthly_hub_grain", "lag", "consecutive_period_filter", "derived_contribution", "per_hub_argmax", "component_share", "top_k"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "grain_safe_multi_fact"],
        "semantic_risks": ["population", "metric_source", "date_boundary", "denominator", "window_cumulative", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "order_fact", "kind": "table", "query": "orders status amount delivery cost cycle production transit", "selected": "orders"},
            {"role": "store_dimension", "kind": "table", "query": "stores hub store id", "selected": "stores"},
            {"role": "hub_dimension", "kind": "table", "query": "hubs hub name", "selected": "hubs"},
            {"role": "cycle_metric", "kind": "column", "table": "orders", "type": "numeric", "query": "order metric cycle time", "selected": "order_metric_cycle_time"},
        ],
        "joins": [
            ["orders", "store_id", "stores", "store_id"],
            ["stores", "hub_id", "hubs", "hub_id"],
        ],
        "mutations": [
            mutation("canceled_orders", "population", "o.order_status = 'FINISHED'", "o.order_status = 'CANCELED'"),
            mutation("nonconsecutive_comparison", "date_boundary", "previous_month = order_month - 1", "previous_month = order_month - 2"),
            mutation("best_improvement_not_deterioration", "ranking", "ORDER BY cycle_deterioration_minutes DESC", "ORDER BY cycle_deterioration_minutes ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_007",
        "database_id": "city_legislation",
        "template_id": "first_same_state_party_switch_by_decade",
        "instruction": (
            "Order every legislator's recorded terms chronologically. A same-state party switch occurs when a "
            "term's party differs from that legislator's immediately previous recorded term while the state remains "
            "the same. Retain only each legislator's first such switch, assign it to the decade in which the new term "
            "started, and aggregate by decade and state. Keep state-decade groups with at least three switching "
            "legislators, then select one state per decade by more switchers, more distinct old-to-new party "
            "directions, and state abbreviation. Report decade, state, switching-legislator count, distinct switch "
            "directions, average years from the legislator's first term to the switch, and average days between the "
            "previous term's end and the switching term's start. Round both averages to two decimals and order by "
            "decade."
        ),
        "sql": """WITH sequenced_terms AS (
    SELECT t.id_bioguide,
           t.state,
           t.party,
           date(t.term_start) AS term_start,
           date(t.term_end) AS term_end,
           MIN(date(t.term_start)) OVER (
               PARTITION BY t.id_bioguide
           ) AS first_term_start,
           LAG(t.party) OVER (
               PARTITION BY t.id_bioguide
               ORDER BY date(t.term_start), t.term_number
           ) AS previous_party,
           LAG(t.state) OVER (
               PARTITION BY t.id_bioguide
               ORDER BY date(t.term_start), t.term_number
           ) AS previous_state,
           LAG(date(t.term_end)) OVER (
               PARTITION BY t.id_bioguide
               ORDER BY date(t.term_start), t.term_number
           ) AS previous_term_end
    FROM legislators_terms AS t
    JOIN legislators AS l ON l.id_bioguide = t.id_bioguide
    WHERE t.party IS NOT NULL
),
party_switches AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY id_bioguide
               ORDER BY term_start
           ) AS switch_sequence
    FROM sequenced_terms
    WHERE previous_party IS NOT NULL
      AND party <> previous_party
      AND state = previous_state
),
first_same_state_switch AS (
    SELECT id_bioguide,
           state,
           previous_party,
           party AS new_party,
           CAST(strftime('%Y', term_start) AS INTEGER) / 10 * 10 AS switch_decade,
           (julianday(term_start) - julianday(first_term_start)) / 365.2425 AS years_to_first_switch,
           julianday(term_start) - julianday(previous_term_end) AS days_between_terms
    FROM party_switches
    WHERE switch_sequence = 1
),
decade_state_switches AS (
    SELECT switch_decade,
           state,
           COUNT(*) AS switching_legislators,
           COUNT(DISTINCT previous_party || '->' || new_party) AS switch_directions,
           AVG(years_to_first_switch) AS avg_years_to_first_switch,
           AVG(days_between_terms) AS avg_days_between_terms
    FROM first_same_state_switch
    GROUP BY switch_decade, state
    HAVING COUNT(*) >= 3
),
ranked_states AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY switch_decade
               ORDER BY switching_legislators DESC,
                        switch_directions DESC,
                        state ASC
           ) AS decade_rank
    FROM decade_state_switches
)
SELECT switch_decade,
       state,
       switching_legislators,
       switch_directions,
       ROUND(avg_years_to_first_switch, 2) AS avg_years_to_first_switch,
       ROUND(avg_days_between_terms, 2) AS avg_days_between_terms
FROM ranked_states
WHERE decade_rank = 1
ORDER BY switch_decade ASC""",
        "expected_columns": ["switch_decade", "state", "switching_legislators", "switch_directions", "avg_years_to_first_switch", "avg_days_between_terms"],
        "row_bounds": [11, 11],
        "operators": ["term_sequence", "lag", "first_event", "derived_decade", "distinct_transition_count", "partitioned_argmax"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "nontrivial_derived_classification"],
        "semantic_risks": ["population", "entity_identity", "date_boundary", "derived_classification", "distinct_dedup", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "term_fact", "kind": "table", "query": "legislators terms party state term start end", "selected": "legislators_terms"},
            {"role": "legislator_dimension", "kind": "table", "query": "legislators bioguide identity", "selected": "legislators"},
            {"role": "party", "kind": "column", "table": "legislators_terms", "type": "text", "query": "party", "selected": "party"},
            {"role": "term_start", "kind": "column", "table": "legislators_terms", "type": "text", "query": "term start", "selected": "term_start"},
        ],
        "joins": [["legislators_terms", "id_bioguide", "legislators", "id_bioguide"]],
        "mutations": [
            mutation("different_state_switches", "population", "state = previous_state", "state <> previous_state"),
            mutation("second_switch", "event_selection", "switch_sequence = 1", "switch_sequence = 2"),
            mutation("fewest_switchers", "ranking", "ORDER BY switching_legislators DESC", "ORDER BY switching_legislators ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_008",
        "database_id": "IPL",
        "template_id": "season_team_phase_run_rate_acceleration",
        "instruction": (
            "For each IPL season, compare each batting team's scoring rate in powerplay overs 1–6 with its scoring "
            "rate in death overs 16–20. For this task, total runs are batsman runs plus all extras aggregated at the "
            "delivery key, and run rate is total runs divided by the number of recorded deliveries times six; do not "
            "remove wides or no-balls from the delivery count. Keep team-seasons with at least 120 recorded deliveries "
            "in each phase. Select one team per eligible season by the largest death-minus-powerplay run-rate "
            "acceleration, then higher death run rate, then smaller team ID. Report season, team, distinct matches "
            "batted, phase delivery counts, both run rates, absolute acceleration, and acceleration as a percentage "
            "of powerplay run rate. Round rates and percentages to two decimals and order by season."
        ),
        "sql": """WITH extras_by_delivery AS (
    SELECT match_id,
           innings_no,
           over_id,
           ball_id,
           SUM(extra_runs) AS extra_runs
    FROM extra_runs
    GROUP BY match_id, innings_no, over_id, ball_id
),
delivery_runs AS (
    SELECT m.season_id,
           b.team_batting AS team_id,
           b.match_id,
           b.innings_no,
           b.over_id,
           b.ball_id,
           bs.runs_scored + COALESCE(e.extra_runs, 0) AS total_runs
    FROM ball_by_ball AS b
    JOIN batsman_scored AS bs
      ON bs.match_id = b.match_id
     AND bs.innings_no = b.innings_no
     AND bs.over_id = b.over_id
     AND bs.ball_id = b.ball_id
    JOIN match AS m ON m.match_id = b.match_id
    LEFT JOIN extras_by_delivery AS e
      ON e.match_id = b.match_id
     AND e.innings_no = b.innings_no
     AND e.over_id = b.over_id
     AND e.ball_id = b.ball_id
),
team_season_phases AS (
    SELECT season_id,
           team_id,
           COUNT(DISTINCT match_id) AS matches_batted,
           SUM(CASE WHEN over_id BETWEEN 1 AND 6 THEN total_runs ELSE 0 END) AS powerplay_runs,
           SUM(CASE WHEN over_id BETWEEN 1 AND 6 THEN 1 ELSE 0 END) AS powerplay_deliveries,
           SUM(CASE WHEN over_id BETWEEN 16 AND 20 THEN total_runs ELSE 0 END) AS death_runs,
           SUM(CASE WHEN over_id BETWEEN 16 AND 20 THEN 1 ELSE 0 END) AS death_deliveries
    FROM delivery_runs
    GROUP BY season_id, team_id
),
qualified_rates AS (
    SELECT *,
           6.0 * powerplay_runs / NULLIF(powerplay_deliveries, 0) AS powerplay_run_rate,
           6.0 * death_runs / NULLIF(death_deliveries, 0) AS death_run_rate,
           6.0 * death_runs / NULLIF(death_deliveries, 0) -
           6.0 * powerplay_runs / NULLIF(powerplay_deliveries, 0) AS acceleration,
           100.0 * (
               6.0 * death_runs / NULLIF(death_deliveries, 0) -
               6.0 * powerplay_runs / NULLIF(powerplay_deliveries, 0)
           ) / NULLIF(6.0 * powerplay_runs / NULLIF(powerplay_deliveries, 0), 0) AS acceleration_pct
    FROM team_season_phases
    WHERE powerplay_deliveries >= 120
      AND death_deliveries >= 120
),
ranked_teams AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY season_id
               ORDER BY acceleration DESC, death_run_rate DESC, team_id ASC
           ) AS season_rank
    FROM qualified_rates
)
SELECT r.season_id,
       t.name AS team_name,
       r.matches_batted,
       r.powerplay_deliveries,
       r.death_deliveries,
       ROUND(r.powerplay_run_rate, 2) AS powerplay_run_rate,
       ROUND(r.death_run_rate, 2) AS death_run_rate,
       ROUND(r.acceleration, 2) AS acceleration,
       ROUND(r.acceleration_pct, 2) AS acceleration_pct
FROM ranked_teams AS r
JOIN team AS t ON t.team_id = r.team_id
WHERE r.season_rank = 1
ORDER BY r.season_id ASC""",
        "expected_columns": ["season_id", "team_name", "matches_batted", "powerplay_deliveries", "death_deliveries", "powerplay_run_rate", "death_run_rate", "acceleration", "acceleration_pct"],
        "row_bounds": [7, 7],
        "operators": ["composite_delivery_key", "preaggregate_extras", "conditional_phase_aggregation", "run_rate", "partitioned_argmax", "ratio"],
        "advanced_families": ["multi_stage_cte", "window", "grain_safe_multi_fact", "nontrivial_derived_classification"],
        "semantic_risks": ["metric_source", "entity_identity", "distinct_dedup", "denominator", "external_rule", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "delivery_fact", "kind": "table", "query": "ball by ball match innings over ball team batting", "selected": "ball_by_ball"},
            {"role": "batsman_runs", "kind": "table", "query": "batsman scored match over ball runs", "selected": "batsman_scored"},
            {"role": "extra_runs", "kind": "table", "query": "extra runs match over ball", "selected": "extra_runs"},
            {"role": "match_dimension", "kind": "table", "query": "match season match id", "selected": "match"},
            {"role": "team_dimension", "kind": "table", "query": "team id name", "selected": "team"},
        ],
        "joins": [
            ["ball_by_ball", "match_id", "batsman_scored", "match_id"],
            ["ball_by_ball", "over_id", "batsman_scored", "over_id"],
            ["ball_by_ball", "ball_id", "batsman_scored", "ball_id"],
            ["ball_by_ball", "match_id", "match", "match_id"],
            ["ball_by_ball", "team_batting", "team", "team_id"],
        ],
        "mutations": [
            mutation("five_over_powerplay", "phase_boundary", "over_id BETWEEN 1 AND 6", "over_id BETWEEN 1 AND 5"),
            mutation("early_death_phase", "phase_boundary", "over_id BETWEEN 16 AND 20", "over_id BETWEEN 15 AND 20"),
            mutation("lowest_acceleration", "ranking", "ORDER BY acceleration DESC", "ORDER BY acceleration ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_009",
        "database_id": "bank_sales_trading",
        "template_id": "sequential_campaign_product_funnel_uplift",
        "instruction": (
            "Evaluate each product covered by a campaign by comparing visit-level sequential funnels during its own "
            "inclusive campaign dates with all dates outside that campaign. A viewing visit starts at the first page "
            "view of the product; it reaches cart only if the same product has an add-to-cart event later in the "
            "visit, and reaches purchase only if a purchase event occurs after that qualifying add. For each period, "
            "calculate distinct product-view journeys, view-to-cart percentage, and view-to-purchase percentage. "
            "Keep products with at least 50 viewing visits both during and outside the campaign, and return the five "
            "largest campaign-minus-outside purchase-rate uplifts, breaking ties by cart-rate uplift and product name. "
            "Report product, category, campaign, both viewing counts, both cart rates and their point uplift, both "
            "purchase rates and their point uplift. Round rates and percentage-point differences to two decimals."
        ),
        "sql": """WITH campaign_product_ranges AS (
    SELECT campaign_name,
           date(start_date) AS start_date,
           date(end_date) AS end_date,
           CAST(substr(products, 1, instr(products, '-') - 1) AS INTEGER) AS first_product_id,
           CAST(substr(products, instr(products, '-') + 1) AS INTEGER) AS last_product_id
    FROM shopping_cart_campaign_identifier
),
product_campaigns AS (
    SELECT p.page_id,
           CAST(p.product_id AS INTEGER) AS product_id,
           p.page_name,
           p.product_category,
           c.campaign_name,
           c.start_date,
           c.end_date
    FROM shopping_cart_page_hierarchy AS p
    JOIN campaign_product_ranges AS c
      ON CAST(p.product_id AS INTEGER) BETWEEN c.first_product_id AND c.last_product_id
),
first_product_views AS (
    SELECT e.visit_id,
           p.page_id,
           p.product_id,
           p.page_name,
           p.product_category,
           p.campaign_name,
           p.start_date,
           p.end_date,
           MIN(e.sequence_number) AS first_view_sequence,
           MIN(e.event_time) AS first_view_time
    FROM shopping_cart_events AS e
    JOIN product_campaigns AS p ON p.page_id = e.page_id
    WHERE e.event_type = 1
    GROUP BY e.visit_id, p.page_id, p.product_id, p.page_name,
             p.product_category, p.campaign_name, p.start_date, p.end_date
),
first_adds_after_view AS (
    SELECT v.visit_id,
           v.product_id,
           MIN(e.sequence_number) AS first_add_sequence
    FROM first_product_views AS v
    JOIN shopping_cart_events AS e
      ON e.visit_id = v.visit_id
     AND e.page_id = v.page_id
     AND e.event_type = 2
     AND e.sequence_number > v.first_view_sequence
    GROUP BY v.visit_id, v.product_id
),
first_purchases AS (
    SELECT visit_id, MIN(sequence_number) AS first_purchase_sequence
    FROM shopping_cart_events
    WHERE event_type = 3
    GROUP BY visit_id
),
visit_product_journeys AS (
    SELECT v.*,
           a.first_add_sequence,
           p.first_purchase_sequence,
           CASE WHEN date(v.first_view_time) BETWEEN v.start_date AND v.end_date
                THEN 'during_campaign' ELSE 'outside_campaign' END AS period_type,
           CASE WHEN a.first_add_sequence IS NOT NULL THEN 1 ELSE 0 END AS reached_cart,
           CASE WHEN p.first_purchase_sequence > a.first_add_sequence THEN 1 ELSE 0 END AS reached_purchase
    FROM first_product_views AS v
    LEFT JOIN first_adds_after_view AS a
      ON a.visit_id = v.visit_id AND a.product_id = v.product_id
    LEFT JOIN first_purchases AS p ON p.visit_id = v.visit_id
),
product_period_funnels AS (
    SELECT product_id,
           page_name,
           product_category,
           campaign_name,
           period_type,
           COUNT(*) AS viewing_visits,
           SUM(reached_cart) AS cart_visits,
           SUM(reached_purchase) AS purchasing_visits,
           100.0 * SUM(reached_cart) / COUNT(*) AS view_to_cart_pct,
           100.0 * SUM(reached_purchase) / COUNT(*) AS view_to_purchase_pct
    FROM visit_product_journeys
    GROUP BY product_id, page_name, product_category, campaign_name, period_type
),
product_comparisons AS (
    SELECT product_id,
           page_name AS product_name,
           product_category,
           campaign_name,
           MAX(CASE WHEN period_type = 'during_campaign' THEN viewing_visits END) AS campaign_viewing_visits,
           MAX(CASE WHEN period_type = 'outside_campaign' THEN viewing_visits END) AS outside_viewing_visits,
           MAX(CASE WHEN period_type = 'during_campaign' THEN view_to_cart_pct END) AS campaign_cart_pct,
           MAX(CASE WHEN period_type = 'outside_campaign' THEN view_to_cart_pct END) AS outside_cart_pct,
           MAX(CASE WHEN period_type = 'during_campaign' THEN view_to_purchase_pct END) AS campaign_purchase_pct,
           MAX(CASE WHEN period_type = 'outside_campaign' THEN view_to_purchase_pct END) AS outside_purchase_pct
    FROM product_period_funnels
    GROUP BY product_id, page_name, product_category, campaign_name
),
ranked_uplift AS (
    SELECT *,
           campaign_purchase_pct - outside_purchase_pct AS purchase_uplift_points,
           campaign_cart_pct - outside_cart_pct AS cart_uplift_points,
           ROW_NUMBER() OVER (
               ORDER BY campaign_purchase_pct - outside_purchase_pct DESC,
                        campaign_cart_pct - outside_cart_pct DESC,
                        product_name ASC
           ) AS uplift_rank
    FROM product_comparisons
    WHERE campaign_viewing_visits >= 50 AND outside_viewing_visits >= 50
)
SELECT product_name,
       product_category,
       campaign_name,
       campaign_viewing_visits,
       outside_viewing_visits,
       ROUND(campaign_cart_pct, 2) AS campaign_cart_pct,
       ROUND(outside_cart_pct, 2) AS outside_cart_pct,
       ROUND(cart_uplift_points, 2) AS cart_uplift_points,
       ROUND(campaign_purchase_pct, 2) AS campaign_purchase_pct,
       ROUND(outside_purchase_pct, 2) AS outside_purchase_pct,
       ROUND(purchase_uplift_points, 2) AS purchase_uplift_points
FROM ranked_uplift
WHERE uplift_rank <= 5
ORDER BY uplift_rank ASC""",
        "expected_columns": ["product_name", "product_category", "campaign_name", "campaign_viewing_visits", "outside_viewing_visits", "campaign_cart_pct", "outside_cart_pct", "cart_uplift_points", "campaign_purchase_pct", "outside_purchase_pct", "purchase_uplift_points"],
        "row_bounds": [5, 5],
        "operators": ["encoded_range_parsing", "ordered_event_funnel", "first_event", "period_counterfactual", "conditional_pivot", "uplift", "top_k"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "grain_safe_multi_fact", "nontrivial_derived_classification"],
        "semantic_risks": ["population", "entity_identity", "distinct_dedup", "date_boundary", "derived_classification", "denominator", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "event_fact", "kind": "table", "query": "shopping cart events visit page event sequence time", "selected": "shopping_cart_events"},
            {"role": "page_dimension", "kind": "table", "query": "shopping cart page hierarchy product category page", "selected": "shopping_cart_page_hierarchy"},
            {"role": "campaign_dimension", "kind": "table", "query": "shopping cart campaign identifier products start end", "selected": "shopping_cart_campaign_identifier"},
            {"role": "sequence", "kind": "column", "table": "shopping_cart_events", "type": "numeric", "query": "sequence number", "selected": "sequence_number"},
        ],
        "joins": [["shopping_cart_events", "page_id", "shopping_cart_page_hierarchy", "page_id"]],
        "mutations": [
            mutation("add_before_view", "event_order", "e.sequence_number > v.first_view_sequence", "e.sequence_number < v.first_view_sequence"),
            mutation("purchase_before_add", "event_order", "p.first_purchase_sequence > a.first_add_sequence", "p.first_purchase_sequence < a.first_add_sequence"),
            mutation("lowest_purchase_uplift", "ranking", "ORDER BY campaign_purchase_pct - outside_purchase_pct DESC", "ORDER BY campaign_purchase_pct - outside_purchase_pct ASC"),
        ],
    },
    {
        "sample_id": "sqlite_hard_010",
        "database_id": "modern_data",
        "template_id": "state_longest_daily_case_decrease_run",
        "instruction": (
            "Derive nonnegative daily new COVID case counts in 2020 from cumulative totals, using only observations "
            "whose state has a record on the immediately previous calendar day and excluding negative corrections. "
            "A decrease is a day whose new-case count is strictly below the preceding valid day's count; missing "
            "dates or a non-decrease break a run. For each state, choose its longest run with at least three "
            "consecutive decreases, breaking ties by the larger drop from the count before the first decrease to the "
            "last count, then earlier start. Report state, run start and end dates (the start is the comparison day's "
            "date before the first decrease), number of decreases, starting and ending daily cases, absolute and "
            "percentage drop, average daily cases on the decreasing days, the state's average across all valid 2020 "
            "daily counts, and the run average as a percentage of that baseline. Round averages and percentages to "
            "two decimals and order by state."
        ),
        "sql": """WITH cumulative_lags AS (
    SELECT state,
           date(date) AS report_date,
           total_cases,
           LAG(date(date)) OVER (
               PARTITION BY state ORDER BY date(date)
           ) AS previous_report_date,
           LAG(total_cases) OVER (
               PARTITION BY state ORDER BY date(date)
           ) AS previous_total_cases
    FROM statistics
    WHERE date >= '2020-01-01' AND date < '2021-01-01'
),
daily_cases AS (
    SELECT state,
           report_date,
           total_cases - previous_total_cases AS new_cases
    FROM cumulative_lags
    WHERE previous_report_date = date(report_date, '-1 day')
      AND total_cases >= previous_total_cases
),
daily_comparisons AS (
    SELECT *,
           LAG(report_date) OVER (
               PARTITION BY state ORDER BY report_date
           ) AS previous_daily_date,
           LAG(new_cases) OVER (
               PARTITION BY state ORDER BY report_date
           ) AS previous_new_cases
    FROM daily_cases
),
decrease_markers AS (
    SELECT *,
           CASE WHEN previous_daily_date = date(report_date, '-1 day')
                     AND new_cases < previous_new_cases
                THEN 1 ELSE 0 END AS is_decrease
    FROM daily_comparisons
),
labeled_runs AS (
    SELECT *,
           SUM(CASE WHEN is_decrease = 0 THEN 1 ELSE 0 END) OVER (
               PARTITION BY state ORDER BY report_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS run_id
    FROM decrease_markers
),
decrease_runs AS (
    SELECT state,
           run_id,
           date(MIN(report_date), '-1 day') AS streak_start_date,
           MAX(report_date) AS streak_end_date,
           COUNT(*) AS consecutive_decreases,
           MAX(previous_new_cases) AS starting_daily_cases,
           MIN(new_cases) AS ending_daily_cases,
           AVG(new_cases) AS avg_daily_cases_during_decreases
    FROM labeled_runs
    WHERE is_decrease = 1
    GROUP BY state, run_id
    HAVING COUNT(*) >= 3
),
state_baselines AS (
    SELECT state, AVG(new_cases) AS state_avg_daily_cases
    FROM daily_cases
    GROUP BY state
),
ranked_runs AS (
    SELECT r.*,
           b.state_avg_daily_cases,
           ROW_NUMBER() OVER (
               PARTITION BY r.state
               ORDER BY r.consecutive_decreases DESC,
                        r.starting_daily_cases - r.ending_daily_cases DESC,
                        r.streak_start_date ASC
           ) AS state_rank
    FROM decrease_runs AS r
    JOIN state_baselines AS b ON b.state = r.state
)
SELECT state,
       streak_start_date,
       streak_end_date,
       consecutive_decreases,
       starting_daily_cases,
       ending_daily_cases,
       starting_daily_cases - ending_daily_cases AS absolute_case_drop,
       ROUND(100.0 * (starting_daily_cases - ending_daily_cases) /
             NULLIF(starting_daily_cases, 0), 2) AS case_drop_pct,
       ROUND(avg_daily_cases_during_decreases, 2) AS avg_daily_cases_during_decreases,
       ROUND(state_avg_daily_cases, 2) AS state_avg_daily_cases,
       ROUND(100.0 * avg_daily_cases_during_decreases /
             NULLIF(state_avg_daily_cases, 0), 2) AS streak_avg_pct_of_state_baseline
FROM ranked_runs
WHERE state_rank = 1
ORDER BY state ASC""",
        "expected_columns": ["state", "streak_start_date", "streak_end_date", "consecutive_decreases", "starting_daily_cases", "ending_daily_cases", "absolute_case_drop", "case_drop_pct", "avg_daily_cases_during_decreases", "state_avg_daily_cases", "streak_avg_pct_of_state_baseline"],
        "row_bounds": [4, 4],
        "operators": ["cumulative_to_daily", "lag", "gap_validation", "gaps_and_islands", "state_baseline", "per_state_argmax", "ratio"],
        "advanced_families": ["multi_stage_cte", "window", "temporal_cohort_or_sequence", "nontrivial_derived_classification"],
        "semantic_risks": ["population", "date_boundary", "window_cumulative", "derived_classification", "denominator", "ranking_tie"],
        "complexity_target": HARD_TARGET,
        "roles": [
            {"role": "cumulative_daily_fact", "kind": "table", "query": "statistics date state total cases deaths", "selected": "statistics"},
            {"role": "cumulative_cases", "kind": "column", "table": "statistics", "type": "numeric", "query": "total cases", "selected": "total_cases"},
            {"role": "report_date", "kind": "column", "table": "statistics", "type": "text", "query": "date", "selected": "date"},
        ],
        "joins": [],
        "mutations": [
            mutation("allow_flat_days", "comparison", "new_cases < previous_new_cases", "new_cases <= previous_new_cases"),
            mutation("five_decreases_required", "population", "HAVING COUNT(*) >= 3", "HAVING COUNT(*) >= 5"),
            mutation("largest_drop_over_longest_run", "ranking", "ORDER BY r.consecutive_decreases DESC", "ORDER BY r.starting_daily_cases - r.ending_daily_cases DESC"),
        ],
    },
]
