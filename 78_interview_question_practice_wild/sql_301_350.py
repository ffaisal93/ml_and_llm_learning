"""PostgreSQL answer strings; portable queries are exercised directly in SQLite."""
SQL={}
SQL[319]='''-- DEMO schema: lots(lot_id,title), bids(bid_id,lot_id,bidder_id,amount).
WITH maxima AS (
 SELECT lot_id,MAX(amount) winning_amount FROM bids GROUP BY lot_id
), winners AS (
 SELECT DISTINCT b.lot_id,b.bidder_id,b.amount
 FROM bids b JOIN maxima m ON b.lot_id=m.lot_id AND b.amount=m.winning_amount
)
SELECT l.lot_id,w.bidder_id,w.amount AS winning_amount
FROM lots l LEFT JOIN winners w ON w.lot_id=l.lot_id
ORDER BY l.lot_id,w.bidder_id;'''
SQL[321]='''WITH totals AS (
 SELECT p.block_type,COUNT(c.block_id) AS total_children
 FROM blocks p LEFT JOIN blocks c
   ON c.workspace_id=p.workspace_id AND c.parent_id=p.block_id
 GROUP BY p.block_type
)
SELECT block_type,total_children FROM totals
WHERE total_children=(SELECT MAX(total_children) FROM totals);'''
SQL[322]='''WITH totals AS (
 SELECT u.country,COUNT(*) AS event_count
 FROM events e JOIN users u ON u.id=e.user_id GROUP BY u.country
)
SELECT country,event_count FROM totals
WHERE event_count=(SELECT MAX(event_count) FROM totals);'''
SQL[323]='''WITH distinct_intervals AS (
 SELECT DISTINCT run_id,start_time,end_time FROM intervals
), previous AS (
 SELECT run_id,start_time,end_time,
   MAX(end_time) OVER(PARTITION BY run_id ORDER BY start_time,end_time
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS max_previous_end
 FROM distinct_intervals
), starts AS (
 SELECT *,CASE WHEN max_previous_end IS NULL OR start_time>max_previous_end THEN 1 ELSE 0 END AS new_group
 FROM previous
), islands AS (
 SELECT *,SUM(new_group) OVER(PARTITION BY run_id ORDER BY start_time,end_time
       ROWS UNBOUNDED PRECEDING) AS grp FROM starts
), merged AS (
 SELECT run_id,grp,MIN(start_time) lo,MAX(end_time) hi FROM islands GROUP BY run_id,grp
)
SELECT run_id,SUM(hi-lo) AS total_non_overlapping_duration FROM merged GROUP BY run_id;'''
SQL[324]='''-- DAU; UTC timestamp-without-time-zone input.
SELECT started_at::date AS day,COUNT(DISTINCT user_id) AS dau
FROM workouts
WHERE started_at>=:start_date::date AND started_at<:end_date::date+INTERVAL '1 day'
GROUP BY started_at::date ORDER BY day;

-- Rolling 30-day MAU, including zero-activity output days.
WITH days AS (
 SELECT generate_series(:start_date::date,:end_date::date,INTERVAL '1 day')::date AS day
), user_days AS (
 SELECT DISTINCT user_id,started_at::date AS active_day FROM workouts
 WHERE started_at>=:start_date::date-INTERVAL '29 days'
   AND started_at<:end_date::date+INTERVAL '1 day'
)
SELECT d.day,COUNT(DISTINCT u.user_id) AS mau
FROM days d LEFT JOIN user_days u ON u.active_day BETWEEN d.day-29 AND d.day
GROUP BY d.day ORDER BY d.day;

-- First-workout cohorts use complete available history, not just the report range.
WITH firsts AS (
 SELECT user_id,date_trunc('week',MIN(started_at))::date AS cohort_week
 FROM workouts GROUP BY user_id
), sizes AS (
 SELECT cohort_week,COUNT(*) AS cohort_size FROM firsts GROUP BY cohort_week
), weekly AS (
 SELECT DISTINCT user_id,date_trunc('week',started_at)::date AS active_week FROM workouts
), retained AS (
 SELECT f.cohort_week,(w.active_week-f.cohort_week)/7 AS week_offset,COUNT(*) AS retained_users
 FROM firsts f JOIN weekly w USING(user_id) GROUP BY f.cohort_week,week_offset
)
SELECT r.cohort_week,r.week_offset,s.cohort_size,r.retained_users,
       ROUND(r.retained_users::numeric/s.cohort_size,2) AS retention_rate
FROM retained r JOIN sizes s USING(cohort_week) ORDER BY r.cohort_week,r.week_offset;

-- Exactly three favourite types at most, deterministic count ties.
WITH counts AS (
 SELECT user_id,workout_type_id,COUNT(*) AS workouts_count
 FROM workouts GROUP BY user_id,workout_type_id
), ranked AS (
 SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY workouts_count DESC,workout_type_id) AS rank
 FROM counts
)
SELECT user_id,workout_type_id,workouts_count,rank FROM ranked WHERE rank<=3 ORDER BY user_id,rank;'''
SQL[330]='''WITH totals AS (
 SELECT event_type,COUNT(*) AS event_count FROM events GROUP BY event_type
)
SELECT event_type,event_count FROM totals
WHERE event_count=(SELECT MAX(event_count) FROM totals);'''
SQL[331]='''-- Chosen schema: listings(id,host_id), listing_stats_daily(listing_id,date,nights_booked,gross_revenue,price_count).
SELECT l.id AS listing_id,COALESCE(SUM(s.nights_booked),0) AS nights_booked,
       SUM(s.gross_revenue)/NULLIF(SUM(s.price_count),0) AS avg_price
FROM listings l LEFT JOIN listing_stats_daily s
 ON s.listing_id=l.id AND s.date>=:start_date AND s.date<=:end_date
WHERE l.host_id=:host_id GROUP BY l.id ORDER BY l.id;'''
SQL[333]='''CREATE TABLE students(id INTEGER PRIMARY KEY,name VARCHAR(255),age INTEGER);
INSERT INTO students VALUES(1,'Alice',22),(2,'Bob',25),(3,'Charlie',30);
SELECT id,name,age FROM students ORDER BY id;
SELECT id,name,age FROM students ORDER BY id LIMIT 2;
SELECT id,name,age FROM students WHERE age>24 ORDER BY id;'''
SQL[334]='''WITH event_totals AS (
 SELECT event_type,COUNT(*) AS event_count FROM events GROUP BY event_type
), top_types AS (
 SELECT event_type FROM event_totals WHERE event_count=(SELECT MAX(event_count) FROM event_totals)
), distribution AS (
 SELECT e.event_type,COALESCE(s.plan_type,'UNKNOWN') AS plan_type,COUNT(*) AS event_count
 FROM events e JOIN top_types t ON (t.event_type=e.event_type OR (t.event_type IS NULL AND e.event_type IS NULL))
 LEFT JOIN users u ON u.id=e.user_id
 LEFT JOIN subscription s ON s.id=u.subscription_id AND s.user_id=u.id
 GROUP BY e.event_type,COALESCE(s.plan_type,'UNKNOWN')
), ranked AS (
 SELECT *,DENSE_RANK() OVER(PARTITION BY event_type ORDER BY event_count DESC) AS rnk FROM distribution
)
SELECT event_type,plan_type,event_count FROM ranked WHERE rnk=1;'''
SQL[335]='''SELECT 100.0*SUM(CASE WHEN NOT EXISTS (
 SELECT 1 FROM events e WHERE e.block_id=b.block_id AND e.event_type='edit_text'
) THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS never_edited_percentage
FROM blocks b;'''
SQL[336]='''-- PostgreSQL, run outside an explicit transaction:
CREATE INDEX CONCURRENTLY job_new_latest
 ON job(created_on DESC,id DESC) WHERE status='new';

-- Atomic claim, one short transaction per batch:
BEGIN;
WITH picked AS (
 SELECT id FROM job WHERE status='new' ORDER BY created_on DESC,id DESC
 FOR UPDATE SKIP LOCKED LIMIT 1
)
UPDATE job SET status='running' FROM picked WHERE job.id=picked.id
RETURNING job.*;
COMMIT;'''
SQL[337]='''WITH combined AS (
 SELECT 'CPU' AS resource_type,application_id,usage_percentage
 FROM cpu_metrics WHERE dt>='2024-02-01 00:00:00' AND dt<'2024-03-01 00:00:00'
 UNION ALL
 SELECT 'Memory' AS resource_type,application_id,usage_percentage
 FROM memory_metrics WHERE dt>='2024-02-01 00:00:00' AND dt<'2024-03-01 00:00:00'
)
SELECT resource_type,application_id,
 TO_CHAR(ROUND(AVG(usage_percentage),2),'FM999990.00') AS average_usage_percentage
FROM combined GROUP BY application_id,resource_type
HAVING AVG(usage_percentage)>50
ORDER BY application_id,resource_type;'''
SQL[339]='''-- DEMO schema employees(employee_id,name), projects(project_id,start_date,end_date,budget,status),
-- assignments(employee_id,project_id), with a unique membership pair.
SELECT e.employee_id,e.name FROM employees e WHERE EXISTS (
 SELECT 1 FROM assignments a JOIN projects p USING(project_id)
 WHERE a.employee_id=e.employee_id AND p.status='completed'
   AND p.end_date::date-p.start_date::date>90
);

WITH membership AS (
 SELECT DISTINCT employee_id,project_id FROM assignments
), sizes AS (
 SELECT project_id,COUNT(*) AS employees FROM membership GROUP BY project_id
), shares AS (
 SELECT m.employee_id,m.project_id,p.budget::numeric/s.employees AS allocated_budget
 FROM membership m JOIN projects p USING(project_id) JOIN sizes s USING(project_id)
)
SELECT employee_id,AVG(allocated_budget) AS average_budget_per_project
FROM shares GROUP BY employee_id;'''
SQL[340]='''WITH days AS (
 SELECT DISTINCT user_id,login_date::date AS day FROM user_login_log WHERE login_date IS NOT NULL
), numbered AS (
 SELECT *,day-(ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY day))::int AS island
 FROM days
), streaks AS (
 SELECT user_id,island,COUNT(*) AS length FROM numbered GROUP BY user_id,island
), longest AS (
 SELECT user_id,MAX(length) AS max_streak FROM streaks GROUP BY user_id
), active AS (
 SELECT user_id,COUNT(*) AS active_days FROM days GROUP BY user_id
)
SELECT l.user_id,l.max_streak,LEAST(l.max_streak,15) AS max_streak_in_15_days,a.active_days
FROM longest l JOIN active a USING(user_id);

SELECT user_id,COUNT(DISTINCT login_date::date) AS active_days
FROM user_login_log GROUP BY user_id HAVING COUNT(DISTINCT login_date::date)>=2;'''
SQL[341]='''WITH totals AS (
 SELECT v.state,v.city,t.vendor_id,
 SUM(CASE WHEN t.transaction_type='PURCHASE' THEN t.amount_charged
          WHEN t.transaction_type='REFUND' THEN -t.amount_charged ELSE 0 END) AS net_amount
 FROM transactions t JOIN vendors v ON v.vendor_id=t.vendor_id
 WHERE t.timestamp>=CURRENT_DATE-INTERVAL '2 years' AND t.timestamp<CURRENT_DATE+INTERVAL '1 day'
 GROUP BY v.state,v.city,t.vendor_id
), ranked AS (
 SELECT *,DENSE_RANK() OVER(PARTITION BY state,city ORDER BY net_amount DESC) AS rank_in_city FROM totals
)
SELECT state,city,vendor_id,net_amount,rank_in_city FROM ranked WHERE rank_in_city<=3
ORDER BY state,city,rank_in_city,vendor_id;'''
SQL[342]='''SELECT vendor_id,
 SUM(CASE WHEN transaction_type='PURCHASE' THEN amount_charged
          WHEN transaction_type='REFUND' THEN -amount_charged ELSE 0 END) AS net_amount
FROM transactions
WHERE timestamp>=CURRENT_DATE-INTERVAL '2 years' AND timestamp<CURRENT_DATE+INTERVAL '1 day'
GROUP BY vendor_id ORDER BY net_amount DESC,vendor_id LIMIT 3;'''
SQL[343]='''SELECT vendor_id,SUM(transaction_dollars) AS total_dollars_charged
FROM transactions
WHERE transaction_type='PURCHASE'
 AND transaction_time>=CURRENT_DATE-INTERVAL '2 years'
 AND transaction_time<CURRENT_DATE+INTERVAL '1 day'
GROUP BY vendor_id ORDER BY total_dollars_charged DESC NULLS LAST,vendor_id LIMIT 3;'''
