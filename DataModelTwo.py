import json
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from settings import app

db = SQLAlchemy(app)

class MachineLearningData2(db.Model):
    __tablename__ = 'machine_learning_data2'
    index = db.Column(db.Integer, primary_key=True)
    headline = db.Column(db.String(80), nullable=False)
    is_sarcastic = db.Column(db.Integer)