import os
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import AsyncOpenAI
import logging

logger = logging.getLogger("db")

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self._llm = AsyncOpenAI(
            api_key=os.getenv("ASI1_API_KEY"),
            base_url="https://api.asi1.ai/v1",
        )
    
    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                database=os.getenv("DB_NAME", "bookshelf"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres")
            )
            register_vector(self.conn)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise e

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    async def _get_embedding(self, text: str) -> list[float]:
        if not text:
            return [0.0] * 1536
        response = await self._llm.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    async def insert_book(self, isbn: str, title: str, author: str, summary: str, status: str):
        if not self.conn:
            self.connect()
            
        embedding = await self._get_embedding(f"{title} by {author}. {summary}")
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO books (isbn, title, author, summary, status, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (isbn) DO UPDATE 
                    SET status = EXCLUDED.status, summary = EXCLUDED.summary, embedding = EXCLUDED.embedding
                    """,
                    (isbn, title, author, summary, status, embedding)
                )
                self.conn.commit()
                logger.info(f"Inserted/Updated book: {title} ({status})")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error inserting book: {e}")
            raise e

    async def search_similar_books(self, query: str, limit: int = 3):
        if not self.conn:
            self.connect()
            
        query_embedding = await self._get_embedding(query)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT title, author, summary, status, 1 - (embedding <=> %s::vector) as similarity
                    FROM books
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, limit)
                )
                results = cur.fetchall()
                return [
                    {
                        "title": row[0],
                        "author": row[1],
                        "summary": row[2],
                        "status": row[3],
                        "similarity": row[4]
                    } for row in results
                ]
        except Exception as e:
            logger.error(f"Error searching similar books: {e}")
            return []

    def check_duplicate_isbn(self, isbn: str) -> dict:
        if not self.conn:
            self.connect()
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, title, status FROM books WHERE isbn = %s", (isbn,))
                result = cur.fetchone()
                if result:
                    return {"id": result[0], "title": result[1], "status": result[2]}
                return None
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return None

    def manual_wipe(self):
        if not self.conn:
            self.connect()
        try:
            with self.conn.cursor() as cur:
                cur.execute("CALL manual_wipe()")
                self.conn.commit()
                logger.info("Database manually wiped")
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error during manual wipe: {e}")
            return False

# Global instance
db_manager = DatabaseManager()
