# models.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///exam_reports.db"  # You can change this to another DB if needed

engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class ExamReport(Base):
    __tablename__ = 'exam_reports'
    id = Column(Integer, primary_key=True, autoincrement=True)
    exam_id = Column(String, unique=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    total_events = Column(Integer, default=0)
    object_detections = Column(JSON, default={})
    voice_candidate = Column(Integer, default=0)
    voice_unknown = Column(Integer, default=0)
    voice_noise = Column(Integer, default=0)
    face_not_found = Column(Integer, default=0)
    multiple_faces = Column(Integer, default=0)
    eye_looking_away = Column(Integer, default=0)
    eye_suspicious_movement = Column(Integer, default=0)
    eye_blinking_violations = Column(Integer, default=0)
    frontend_tab_switches = Column(Integer, default=0)
    frontend_fullscreen_exits = Column(Integer, default=0)
    frontend_inactivity = Column(Integer, default=0)
    frontend_extended_monitor = Column(Integer, default=0)
    frontend_geolocation_error = Column(Integer, default=0)

Base.metadata.create_all(engine)

def add_exam_report(report):
    session = SessionLocal()
    try:
        exam_report = ExamReport(
            exam_id=report['exam_id'],
            start_time=datetime.datetime.fromisoformat(report['start_time']),
            end_time=datetime.datetime.fromisoformat(report['end_time']),
            total_events=report['summary']['total_events'],
            object_detections=report['summary']['object_detections'],
            voice_candidate=report['summary']['voice_events']['candidate'],
            voice_unknown=report['summary']['voice_events']['unknown'],
            voice_noise=report['summary']['voice_events']['noise'],
            face_not_found=report['summary']['face_warnings']['face_not_found'],
            multiple_faces=report['summary']['face_warnings']['multiple_faces'],
            eye_looking_away=report['summary']['eye_warnings']['looking_away'],
            eye_suspicious_movement=report['summary']['eye_warnings']['suspicious_movement'],
            eye_blinking_violations=report['summary']['eye_warnings']['blinking_violations'],
            frontend_tab_switches=report['summary']['frontend_violations']['tab_switches'],
            frontend_fullscreen_exits=report['summary']['frontend_violations']['fullscreen_exits'],
            frontend_inactivity=report['summary']['frontend_violations']['inactivity'],
            frontend_extended_monitor=report['summary']['frontend_violations']['extended_monitor'],
            frontend_geolocation_error=report['summary']['frontend_violations']['geolocation_error']
        )
        session.add(exam_report)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
