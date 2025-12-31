CREATE TABLE aurora_login_events (
    event_date       DATE        NOT NULL,
    event_timestamp  TIMESTAMP   NOT NULL,
    user_id          VARCHAR(64) NOT NULL,
    event_type       VARCHAR(32) NOT NULL,
    device_type      VARCHAR(32),
    region           VARCHAR(16),
    source_system    VARCHAR(64) NOT NULL,
    ingestion_ts     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
