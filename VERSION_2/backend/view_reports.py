import json
from database import SessionLocal
from models import Report

session = SessionLocal()
reports = session.query(Report).all()

for report in reports:
    print(f"\nID: {report.id}")
    print(f"Title: {report.title}")
    print("Content:")

    try:
        # Try to parse as JSON for pretty print
        content = json.loads(report.content)
        print(json.dumps(content, indent=4))
    except json.JSONDecodeError:
        print(report.content)

    print(f"Created At: {report.created_at}")
    print("-" * 50)

session.close()
