from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import uuid

app = FastAPI()

class Question:
  questionId: str
  questionText: str
  answerText: str
  def __init__(self, qId: str, qText: str,aText: str) -> None:
        self.questionId = qId
        self.questionText = qText
        self.answerText = aText
  

@app.get("/")
def hello():
  return {"Hello Everyone!"}


@app.get("/questions")
async def getQuestions():
  questions=[]
  df = pd.read_csv("alzheimer_questionaire.csv",encoding = 'cp1252')
  df=df.sample(5)
  for i, row in df.iterrows():
    print(row['question'],row['answer'] )
    questions.append(Question(uuid.uuid4().hex,row['question'], row['answer']))
  return questions
