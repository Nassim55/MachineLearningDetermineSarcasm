import json
import pandas as pd
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from settings import app

db = SQLAlchemy(app)

class MachineLearningData(db.Model):
    __tablename__ = 'machine_learning_data'
    index = db.Column(db.Integer, primary_key=True)
    headline = db.Column(db.String(80), nullable=False)
    is_sarcastic = db.Column(db.Integer)

    def json(self):
        return { 'headline': self.headline, 'is_sarcastic': self.is_sarcastic }

    def get_all_data():
        return [MachineLearningData.json(ml_data) for ml_data in MachineLearningData.query.all()]

    def get_all_data_json():
        ml_data_list = [MachineLearningData.json(ml_data) for ml_data in MachineLearningData.query.all()]
        ml_data_dict = {'ml_data': ml_data_list}
        with open('ml_data.json', 'w') as outfile:
            json.dump(ml_data_dict, outfile)
        return 'done'
