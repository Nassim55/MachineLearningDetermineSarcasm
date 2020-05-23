import json
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine 
from settings import app

db = SQLAlchemy(app)

# with open('Sarcasm_Headlines_Dataset.json', 'r') as f:
#     datastore = json.loads("[" + 
#         f.read().replace("}\n{", "},\n{") + 
#     "]")

# df = pd.DataFrame(datastore)
# del df['article_link']

class MachineLearningData(db.Model):
    __tablename__ = 'machine_learning_data'
    id = db.Column(db.Integer, primary_key=True)
    headline = db.Column(db.String(80), nullable=False)
    is_sarcastic = db.Column(db.Integer)

    # CODE TO CREATE THE DATABASE INITIALLY
    #engine = create_engine('sqlite:///database.db')
    #df.to_sql('machine_learning_data', engine, if_exists='replace')

    def json(self):
        return {'headline': self.headline, 'is_sarcastic': self.is_sarcastic}    