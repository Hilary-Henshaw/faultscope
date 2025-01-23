-- FaultScope TimescaleDB Schema
-- Initializes all tables, hypertables, indexes, and continuous aggregates.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ─────────────────────────────────────────────────────────────────────────────
-- Core registry tables
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS machines (
    machine_id      TEXT        PRIMARY KEY,
    machine_type    TEXT        NOT NULL CHECK (
        machine_type IN ('turbofan', 'pump', 'compressor', 'motor')
    ),
    location        TEXT        NOT NULL DEFAULT 'unknown',
    status          TEXT        NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'maintenance', 'decommissioned')
    ),
    commissioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB       NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_machines_status
    ON machines (status);

-- ─────────────────────────────────────────────────────────────────────────────
-- Time-series tables (converted to hypertables)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sensor_readings (
    recorded_at     TIMESTAMPTZ NOT NULL,
    machine_id      TEXT        NOT NULL REFERENCES machines (machine_id),
    cycle           INTEGER,
    readings        JSONB       NOT NULL,
    operational     JSONB       NOT NULL DEFAULT '{}',
    quality_flags   TEXT[]      NOT NULL DEFAULT '{}',
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'sensor_readings', 'recorded_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'sensor_readings',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_machine
    ON sensor_readings (machine_id, recorded_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS computed_features (
    computed_at     TIMESTAMPTZ NOT NULL,
    machine_id      TEXT        NOT NULL REFERENCES machines (machine_id),
    window_s        INTEGER     NOT NULL,
    temporal        JSONB       NOT NULL DEFAULT '{}',
    spectral        JSONB       NOT NULL DEFAULT '{}',
    correlation     JSONB       NOT NULL DEFAULT '{}',
    feature_version TEXT        NOT NULL DEFAULT 'v1'
);

SELECT create_hypertable(
    'computed_features', 'computed_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'computed_features',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_computed_features_machine
    ON computed_features (machine_id, computed_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_at     TIMESTAMPTZ NOT NULL,
    machine_id      TEXT        NOT NULL,
    feature_vector  JSONB       NOT NULL,
    rul_cycles      INTEGER,
    health_label    TEXT        CHECK (
        health_label IN (
            'healthy', 'degrading', 'critical', 'imminent_failure'
        )
    ),
    dataset_version TEXT        NOT NULL DEFAULT 'v1',
    split           TEXT        NOT NULL DEFAULT 'train' CHECK (
        split IN ('train', 'validation', 'test')
    )
);

SELECT create_hypertable(
    'feature_snapshots', 'snapshot_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_predictions (
    predicted_at        TIMESTAMPTZ NOT NULL,
    machine_id          TEXT        NOT NULL REFERENCES machines (machine_id),
    rul_cycles          REAL,
    rul_hours           REAL,
    rul_lower_bound     REAL,
    rul_upper_bound     REAL,
    health_label        TEXT,
    health_probabilities JSONB      NOT NULL DEFAULT '{}',
    anomaly_score       REAL,
    confidence          REAL,
    rul_model_version   TEXT,
    health_model_version TEXT,
    latency_ms          INTEGER
);

SELECT create_hypertable(
    'model_predictions', 'predicted_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'model_predictions',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_predictions_machine
    ON model_predictions (machine_id, predicted_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Alert & incident tables
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS detection_rules (
    rule_id         TEXT        PRIMARY KEY,
    rule_name       TEXT        NOT NULL UNIQUE,
    description     TEXT        NOT NULL DEFAULT '',
    severity        TEXT        NOT NULL CHECK (
        severity IN ('info', 'warning', 'critical')
    ),
    condition_type  TEXT        NOT NULL,
    thresholds      JSONB       NOT NULL DEFAULT '{}',
    cooldown_s      INTEGER     NOT NULL DEFAULT 3600,
    enabled         BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id     TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    rule_id         TEXT        NOT NULL REFERENCES detection_rules (rule_id),
    machine_id      TEXT        NOT NULL REFERENCES machines (machine_id),
    severity        TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    details         JSONB       NOT NULL DEFAULT '{}',
    status          TEXT        NOT NULL DEFAULT 'open' CHECK (
        status IN ('open', 'acknowledged', 'closed')
    ),
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    acknowledged_by TEXT,
    resolution_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_machine_status
    ON incidents (machine_id, status, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_severity
    ON incidents (severity, status);

-- ─────────────────────────────────────────────────────────────────────────────
-- Service records & MLOps tables
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS service_records (
    record_id       TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    machine_id      TEXT        NOT NULL REFERENCES machines (machine_id),
    service_type    TEXT        NOT NULL,
    performed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    performed_by    TEXT        NOT NULL DEFAULT 'system',
    notes           TEXT        NOT NULL DEFAULT '',
    rul_before      INTEGER,
    rul_after       INTEGER,
    metadata        JSONB       NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_catalog (
    catalog_id      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    mlflow_run_id   TEXT        UNIQUE,
    model_type      TEXT        NOT NULL CHECK (
        model_type IN ('lifespan_predictor', 'condition_classifier')
    ),
    version_tag     TEXT        NOT NULL,
    stage           TEXT        NOT NULL DEFAULT 'staging' CHECK (
        stage IN ('staging', 'production', 'archived')
    ),
    mae             REAL,
    rmse            REAL,
    r2_score        REAL,
    f1_score        REAL,
    artifact_path   TEXT,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_model_catalog_type_stage
    ON model_catalog (model_type, stage);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id          TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    trigger_reason  TEXT        NOT NULL DEFAULT 'scheduled',
    status          TEXT        NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'running', 'completed', 'failed')
    ),
    dataset_version TEXT        NOT NULL,
    config_snapshot JSONB       NOT NULL DEFAULT '{}',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS drift_events (
    event_id        TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drift_type      TEXT        NOT NULL CHECK (
        drift_type IN ('data', 'concept', 'covariate')
    ),
    affected_features TEXT[]   NOT NULL DEFAULT '{}',
    ks_statistic    REAL,
    p_value         REAL,
    triggered_retrain BOOLEAN  NOT NULL DEFAULT FALSE,
    details         JSONB       NOT NULL DEFAULT '{}'
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Continuous aggregates for dashboard queries
-- ─────────────────────────────────────────────────────────────────────────────

CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_1m
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 minute', recorded_at) AS bucket,
        machine_id,
        avg((readings->>'temperature')::REAL)  AS avg_temperature,
        avg((readings->>'vibration')::REAL)    AS avg_vibration,
        avg((readings->>'pressure')::REAL)     AS avg_pressure,
        count(*)                               AS reading_count
    FROM sensor_readings
    GROUP BY bucket, machine_id
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_1h
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', recorded_at) AS bucket,
        machine_id,
        avg((readings->>'temperature')::REAL)  AS avg_temperature,
        avg((readings->>'vibration')::REAL)    AS avg_vibration,
        avg((readings->>'pressure')::REAL)     AS avg_pressure,
        min((readings->>'vibration')::REAL)    AS min_vibration,
        max((readings->>'vibration')::REAL)    AS max_vibration,
        count(*)                               AS reading_count
    FROM sensor_readings
    GROUP BY bucket, machine_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'sensor_readings_1m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
    'sensor_readings_1h',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Seed default detection rules
-- ─────────────────────────────────────────────────────────────────────────────

INSERT INTO detection_rules
    (rule_id, rule_name, description, severity, condition_type,
     thresholds, cooldown_s, enabled)
VALUES
    ('rul_critical',  'RUL Critical',
     'Remaining useful life below critical threshold',
     'critical', 'rul_below',
     '{"threshold": 10}', 3600, TRUE),

    ('rul_warning',   'RUL Warning',
     'Remaining useful life approaching critical zone',
     'warning', 'rul_below',
     '{"threshold": 30}', 7200, TRUE),

    ('rul_info',      'RUL Informational',
     'Remaining useful life entering watch zone',
     'info', 'rul_below',
     '{"threshold": 50}', 14400, TRUE),

    ('anomaly_critical', 'Anomaly Score Critical',
     'Sensor anomaly score exceeds critical level',
     'critical', 'anomaly_score_above',
     '{"threshold": 0.9}', 1800, TRUE),

    ('anomaly_warning', 'Anomaly Score Warning',
     'Sensor anomaly score elevated',
     'warning', 'anomaly_score_above',
     '{"threshold": 0.7}', 3600, TRUE),

    ('health_imminent', 'Imminent Failure Detected',
     'Health classifier predicts imminent failure',
     'critical', 'health_label_is',
     '{"label": "imminent_failure"}', 1800, TRUE),

    ('health_critical', 'Critical Health Status',
     'Health classifier reports critical condition',
     'warning', 'health_label_is',
     '{"label": "critical"}', 3600, TRUE),

    ('rapid_degradation', 'Rapid RUL Degradation',
     'RUL declining faster than expected rate',
     'critical', 'rul_drop_rate',
     '{"cycles_per_hour": 5}', 3600, TRUE),

    ('multi_sensor_anomaly', 'Multi-Sensor Anomaly',
     'Multiple sensors simultaneously show anomalous readings',
     'critical', 'multi_sensor',
     '{"min_sensors": 3}', 1800, TRUE)
ON CONFLICT (rule_id) DO NOTHING;
