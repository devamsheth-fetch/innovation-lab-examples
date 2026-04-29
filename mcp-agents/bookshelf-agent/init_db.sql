-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the books table
CREATE TABLE IF NOT EXISTS books (
    id SERIAL PRIMARY KEY,
    isbn VARCHAR(20) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL,
    summary TEXT,
    status VARCHAR(50) CHECK (status IN ('read', 'wishlist')) NOT NULL,
    embedding vector(1536), -- Dimension for OpenAI/ASI1 embeddings (e.g. text-embedding-3-small)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for semantic search (IVFFlat or HNSW)
-- Using HNSW for better performance on larger datasets
CREATE INDEX ON books USING hnsw (embedding vector_cosine_ops);

-- Manual Wipe Function
CREATE OR REPLACE PROCEDURE manual_wipe()
LANGUAGE plpgsql
AS $$
BEGIN
    TRUNCATE TABLE books RESTART IDENTITY CASCADE;
    RAISE NOTICE 'Bookshelf database has been manually wiped.';
END;
$$;
