# models/tables.py
from models.db import db

class User(db.Model):
    __tablename__ = 'user'
    email = db.Column(db.String(100), primary_key=True)

class Mail(db.Model):
    __tablename__ = 'mails'
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), primary_key=True)
    mail_id = db.Column(db.String(255), primary_key=True)
    subject = db.Column(db.Text)
    from_ = db.Column("from", db.String(255))
    body = db.Column(db.Text)
    raw_message = db.Column(db.Text)
    date = db.Column(db.DateTime)

class Todo(db.Model):
    __tablename__ = 'todo'
    todo_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    date = db.Column(db.Date, nullable=False)
    event = db.Column(db.Text)
