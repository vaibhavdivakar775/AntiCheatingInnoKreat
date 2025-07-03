from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# Change this to a different URI if needed
DATABASE_URL = "sqlite:///report_database.db"

# Create engine and session
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# Create tables (only runs if tables don't exist)
def init_db():
    Base.metadata.create_all(engine)
