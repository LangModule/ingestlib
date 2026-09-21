-- Read-only role: a person or tool connects as ingestlib_ro to READ the
-- registry, never write. ingestlib itself uses the owner role
-- (POSTGRES_USER=ingestlib) for read-write. Idempotent — safe to re-run on a
-- Postgres you manage yourself. The password here is a local-dev default;
-- change it for any real deployment.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ingestlib_ro') THEN
        CREATE ROLE ingestlib_ro LOGIN PASSWORD 'ro_pw';
    END IF;
END
$$;
GRANT CONNECT ON DATABASE ingestlib TO ingestlib_ro;
GRANT USAGE ON SCHEMA public TO ingestlib_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ingestlib_ro;
-- Alembic creates the tables later (as ingestlib); this makes those future
-- tables readable by ingestlib_ro without a re-grant.
ALTER DEFAULT PRIVILEGES FOR ROLE ingestlib IN SCHEMA public GRANT SELECT ON TABLES TO ingestlib_ro;
