-- W228, PostgreSQL. Group daily first so duplicate dates have deterministic totals.
WITH daily AS (
  SELECT store_id, CAST(date AS date) AS day, SUM(amount) AS amount
  FROM sales GROUP BY store_id, CAST(date AS date)
)
SELECT store_id, day AS date,
       SUM(amount) OVER (PARTITION BY store_id,date_trunc('month',day)
                         ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total_amount
FROM daily ORDER BY store_id,day;

-- W228 hierarchy: manager_id=1 is the root ID, not an employee row filter at every depth.
WITH RECURSIVE reports AS (
  SELECT employee_id,manager_id,name,1 AS level,ARRAY[1,employee_id] AS hierarchy_path
  FROM organization WHERE manager_id=1 AND employee_id<>1
  UNION ALL
  SELECT e.employee_id,e.manager_id,e.name,r.level+1,r.hierarchy_path||e.employee_id
  FROM organization e JOIN reports r ON e.manager_id=r.employee_id
  WHERE NOT e.employee_id=ANY(r.hierarchy_path)
)
SELECT employee_id,manager_id,name,level,hierarchy_path FROM reports ORDER BY hierarchy_path;

-- W228 one scan, explicit NULL-to-zero monetary behavior.
SELECT CAST(date AS date) AS date,
       SUM(CASE WHEN status='completed' THEN COALESCE(amount,0) ELSE 0 END) AS completed_revenue,
       SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending_count,
       SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled_count
FROM orders GROUP BY CAST(date AS date) ORDER BY date;

-- W230 DEMONSTRATION schema: events(user_id,event_id,event_time,kind,value).
-- N=:n, qualifying condition kind='purchase'; preceding means previous event OF ANY kind.
WITH all_events AS (
  SELECT *,LAG(event_time) OVER(PARTITION BY user_id ORDER BY event_time,event_id) AS previous_time,
           LAG(value) OVER(PARTITION BY user_id ORDER BY event_time,event_id) AS previous_value
  FROM events
), qualifying AS (
  SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY event_time DESC,event_id DESC) AS rn
  FROM all_events WHERE kind='purchase'
)
SELECT user_id,event_id,event_time,previous_time,
       event_time-previous_time AS elapsed,value-previous_value AS value_delta
FROM qualifying WHERE rn=:n;
