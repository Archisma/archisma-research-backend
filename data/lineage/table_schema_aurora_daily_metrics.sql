CREATE TABLE aurora_daily_metrics (
    metric_date   DATE        NOT NULL,
    metric_name   VARCHAR(64) NOT NULL,
    metric_value  BIGINT      NOT NULL,
    region        VARCHAR(16),
    created_ts    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (metric_date, metric_name, region)
);
