-- Rocky AI Database Initialization Script
-- This script sets up the initial database schema and data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database if it doesn't exist (this will be handled by Docker)
-- CREATE DATABASE rocky_ai;

-- Connect to rocky_ai database
\c rocky_ai;

-- Create custom types
CREATE TYPE analysis_status AS ENUM ('pending', 'running', 'completed', 'failed');
CREATE TYPE analysis_language AS ENUM ('python', 'r');
CREATE TYPE analysis_type AS ENUM ('descriptive', 'inferential', 'predictive', 'exploratory');

-- Create indexes for better performance
-- These will be created automatically by SQLAlchemy, but we can add custom ones here

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE rocky_ai TO rocky;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rocky;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rocky;

-- Create initial admin user (password: admin123 - change in production)
-- This will be handled by the application, but we can set up the structure here

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, tags) VALUES
('system_startup', 1, 'count', '{"component": "database", "action": "initialization"}'),
('database_version', 15, 'version', '{"component": "postgresql"}');

-- Create a view for analysis statistics
CREATE VIEW analysis_stats AS
SELECT 
    analysis_type,
    language,
    status,
    COUNT(*) as count,
    AVG(execution_time) as avg_execution_time,
    AVG(memory_used) as avg_memory_used,
    DATE_TRUNC('day', created_at) as date
FROM analyses 
GROUP BY analysis_type, language, status, DATE_TRUNC('day', created_at);

-- Create a view for user activity
CREATE VIEW user_activity AS
SELECT 
    u.username,
    u.email,
    COUNT(a.id) as total_analyses,
    COUNT(CASE WHEN a.status = 'completed' THEN 1 END) as completed_analyses,
    COUNT(CASE WHEN a.status = 'failed' THEN 1 END) as failed_analyses,
    AVG(a.execution_time) as avg_execution_time,
    MAX(a.created_at) as last_analysis
FROM users u
LEFT JOIN analyses a ON u.id = a.user_id
GROUP BY u.id, u.username, u.email;

-- Create indexes for better query performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analyses_status ON analyses(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analyses_correlation_id ON analyses(correlation_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_cache_expires_at ON model_cache(expires_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create a function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete expired cache entries
    DELETE FROM model_cache WHERE expires_at < NOW();
    
    -- Delete old system metrics (older than 30 days)
    DELETE FROM system_metrics WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Log cleanup
    INSERT INTO system_metrics (metric_name, metric_value, metric_unit, tags)
    VALUES ('cleanup_executed', 1, 'count', '{"component": "maintenance"}');
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to run cleanup (requires pg_cron extension)
-- This would be set up in production with pg_cron
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

-- Insert initial configuration
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, tags) VALUES
('database_initialized', 1, 'count', '{"component": "database", "action": "initialization", "timestamp": "' || NOW() || '"}');

COMMIT;
