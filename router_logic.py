import random
import psycopg
from psycopg.rows import dict_row

DB_URI = "postgresql://rag_user:rag_password@localhost:5433/rag_evals"

def setup_router_db():
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Create the stats table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS thompson_stats (
                    strategy VARCHAR(50) PRIMARY KEY,
                    alpha INT DEFAULT 1,
                    beta INT DEFAULT 1
                )
            """)
            # Insert base values (1,1) for uniform exploration
            cur.execute("INSERT INTO thompson_stats (strategy) VALUES ('fast_route') ON CONFLICT DO NOTHING")
            cur.execute("INSERT INTO thompson_stats (strategy) VALUES ('smart_route') ON CONFLICT DO NOTHING")

def get_best_route() -> str:
    setup_router_db()
    with psycopg.connect(DB_URI, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT strategy, alpha, beta FROM thompson_stats")
            stats = cur.fetchall()
            
    best_strategy = "smart_route"
    highest_sample = -1
    
    # Thompson Sampling: Draw a random sample from the Beta distribution of each strategy
    for row in stats:
        sample = random.betavariate(row["alpha"], row["beta"])
        if sample > highest_sample:
            highest_sample = sample
            best_strategy = row["strategy"]
            
    return best_strategy

def log_feedback(strategy: str, success: bool):
    """Updates the alpha (success) or beta (failure) count in PostgreSQL"""
    setup_router_db()
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            if success:
                cur.execute("UPDATE thompson_stats SET alpha = alpha + 1 WHERE strategy = %s", (strategy,))
            else:
                cur.execute("UPDATE thompson_stats SET beta = beta + 1 WHERE strategy = %s", (strategy,))