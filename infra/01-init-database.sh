#!/bin/bash

# Load environment variables from .env file
DB_NAME="${POSTGRES_DB:-twitter_db}"
SUPER_USER="${POSTGRES_USER:-postgres}"
SUPER_PASSWORD="${POSTGRES_PASSWORD:-}"

APP_USER="${POSTGRES_TWITTER_USER:-twitter_user}"
APP_PASSWORD="${POSTGRES_TWITTER_PASSWORD:-twitter_password}"

echo "Starting the database: $DB_NAME"
echo "Creating user: $APP_USER"

# Execute PostgreSQL commands
export PGPASSWORD="$SUPER_PASSWORD"

psql -v ON_ERROR_STOP=1 --username "$SUPER_USER" --dbname "$DB_NAME" <<-EOSQL
    -- Create the user if it doesn't exist
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$APP_USER') THEN
            CREATE USER $APP_USER WITH 
                PASSWORD '$APP_PASSWORD'
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                INHERIT
                LOGIN;
        END IF;
    END
    \$\$;
    
    -- Grant CONNECT privileges on the database
    GRANT CONNECT ON DATABASE $DB_NAME TO $APP_USER;
    
    -- Grant permissions on the public schema
    \c $DB_NAME;
    GRANT USAGE ON SCHEMA public TO $APP_USER;
    GRANT CREATE ON SCHEMA public TO $APP_USER;
    
    -- Grant basic privileges (read and write) for future objects
    ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO $APP_USER;
    
    ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT USAGE, SELECT ON SEQUENCES TO $APP_USER;
    
    ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT EXECUTE ON FUNCTIONS TO $APP_USER;
EOSQL

echo "Initial database configuration completed."