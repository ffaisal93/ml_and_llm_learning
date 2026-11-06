# SQL lab

These small examples complement F171–F179. They use SQLite syntax so the checks can run with Python's standard library; date arithmetic and interval literals differ in other databases. Fix an explicit as-of boundary instead of relying on the computer's current date.

## Tables and conventions

The examples assume these columns:

```sql
CREATE TABLE transactions (
  id INTEGER PRIMARY KEY, vendor_id TEXT, day TEXT, net_amount REAL
);
CREATE TABLE logins (user_id TEXT, day TEXT);
CREATE TABLE events (event_type TEXT, plan TEXT);
CREATE TABLE predictions (id INTEGER PRIMARY KEY, entity TEXT, at_time INTEGER);
CREATE TABLE features (entity TEXT, available_at INTEGER, value REAL);
CREATE TABLE impressions (id INTEGER PRIMARY KEY, category TEXT);
CREATE TABLE clicks (impression_id INTEGER);
```

`transactions.net_amount` already uses positive charges and negative refunds in one currency. Dates are ISO `YYYY-MM-DD` strings; logins have already been assigned to the intended timezone. Feature availability uses integer timestamps in one common unit. CTR here means the fraction of impressions with at least one attributed click, rather than raw click count per impression.

## F171: top three vendors by net charge

```sql
SELECT vendor_id, SUM(net_amount) AS total
FROM transactions
WHERE day >= :start_day AND day < :end_day
GROUP BY vendor_id
ORDER BY total DESC, vendor_id ASC
LIMIT 3;
```

This returns exactly three rows if at least three vendors qualify. The secondary key makes ties deterministic. Choose `start_day` as the appropriate calendar boundary two years before the as-of boundary; do not replace that decision casually with 730 days. Filter and convert amounts before aggregation if the schema stores charges and refunds differently.

## F172: distinct-day login streaks

```sql
WITH days AS (
  SELECT DISTINCT user_id, day FROM logins
), numbered AS (
  SELECT user_id, day,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS rn
  FROM days
), grouped AS (
  SELECT user_id, day, date(day, '-' || rn || ' days') AS run_key
  FROM numbered
)
SELECT user_id, MIN(day) AS first_day, MAX(day) AS last_day,
       COUNT(*) AS streak_length
FROM grouped
GROUP BY user_id, run_key
ORDER BY user_id, first_day;
```

For consecutive dates, subtracting their consecutive row numbers gives a constant date. Deduplication must happen before numbering.

## F173: most common event, then most common plan inside it

```sql
WITH event_counts AS (
  SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type
), chosen_event AS (
  SELECT event_type FROM event_counts
  ORDER BY n DESC, event_type ASC LIMIT 1
), plan_counts AS (
  SELECT plan, COUNT(*) AS n
  FROM events
  WHERE event_type = (SELECT event_type FROM chosen_event)
  GROUP BY plan
)
SELECT plan, n FROM plan_counts
ORDER BY n DESC, plan ASC LIMIT 1;
```

This chooses one event and one plan, breaking ties alphabetically. A tie-preserving interpretation requires a different ranking rule. Decide how null categories count before using this on real data.

## F176: point-in-time feature join

```sql
WITH candidates AS (
  SELECT p.id, p.entity, p.at_time, f.value,
         ROW_NUMBER() OVER (
           PARTITION BY p.id ORDER BY f.available_at DESC
         ) AS rn
  FROM predictions AS p
  LEFT JOIN features AS f
    ON f.entity = p.entity AND f.available_at <= p.at_time
)
SELECT id, entity, at_time, value
FROM candidates WHERE rn = 1 ORDER BY id;
```

Use a unique `(entity, available_at)` convention or add a version tie-break. The left join preserves predictions that have no available feature; their value is null. For large workloads, inspect execution plans and suitable indexes instead of materializing an unrestricted history join.

## F179: category CTR without duplicate-click inflation

```sql
SELECT i.category, COUNT(*) AS impressions,
       SUM(CASE WHEN EXISTS (
         SELECT 1 FROM clicks AS c WHERE c.impression_id = i.id
       ) THEN 1 ELSE 0 END) AS clicked_impressions,
       AVG(CASE WHEN EXISTS (
         SELECT 1 FROM clicks AS c WHERE c.impression_id = i.id
       ) THEN 1.0 ELSE 0.0 END) AS ctr
FROM impressions AS i
GROUP BY i.category ORDER BY i.category;
```

`EXISTS` prevents multiple clicks from multiplying one impression. Apply the agreed eligibility and attribution windows before this aggregation. Indexing clicks by `impression_id` can help. Report sample counts along with CTR.

## Run the examples

```bash
python3 -m unittest discover -s 78_interview_question_practice_wild -p 'test_sql_examples.py' -v
```

The checks execute the actual SQL blocks from this page against a small in-memory database. They check net refunds and time boundaries, duplicate login days, conditional plan popularity, point-in-time feature availability and duplicate-click handling.
