# test_postgres.py
from chunkdup import Memory
from chunkdup.postgres_repository import PostgresRepository

def test_postgres():
    # Connect to PostgreSQL
    repo = PostgresRepository("postgresql://postgres:aditya123@localhost:5433/chunkdup")
    memory = Memory(store="postgres", repository=repo)  # or just pass connection_url

    # Add memories
    memory.remember("I use Python for backend development")
    memory.remember("I work at Google as a software engineer")
    memory.remember("I use Python")
    memory.remember("I now use Go")

    # Retrieve
    results = memory.retrieve("What language?")
    print("Retrieved:", results)

    # Get all
    all_memories = memory.get_all()
    print("All memories:", all_memories)

    # Get history
    history = memory.get_history("programming_language")
    print("History:", history)

if __name__ == "__main__":
    test_postgres()