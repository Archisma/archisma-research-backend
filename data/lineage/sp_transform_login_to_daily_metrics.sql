INSERT INTO aurora_daily_metrics (
    metric_date,
    metric_name,
    metric_value,
    region
)
SELECT
    event_date,
    'daily_active_users',
    COUNT(DISTINCT user_id),
    region
FROM aurora_login_events
WHERE event_type = 'login'
GROUP BY event_date, region;
