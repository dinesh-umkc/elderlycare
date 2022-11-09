from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import uuid

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

app = FastAPI()

class Question():
  questionId: str
  questionText: str
  answerText: str
  def __init__(self, qId: str, qText: str,aText: str) -> None:
        self.questionId = qId
        self.questionText = qText
        self.answerText = aText

class Answer(BaseModel):
  questionId: str
  questionText: str
  answerText: str


@app.get("/questions")
async def getQuestions():
  questions=[]
  df = pd.read_csv("alzheimer_questionaire.csv",encoding = 'cp1252')
  df=df.sample(5)
  for i, row in df.iterrows():
    print(row['question'],row['answer'] )
    questions.append(Question(uuid.uuid4().hex,row['question'], row['answer']))
  return questions

@app.post("/answers")
def computeScore(answers: List[Answer]):
  model = SentenceTransformer('bert-base-nli-mean-tokens')
  score=0
  for answer in answers:
    questionEcoding = model.encode(answer.questionText)
    answerEcoding = model.encode(answer.answerText)
    sim_arr=cosine_similarity([questionEcoding],[answerEcoding])
    print('question:',answer.questionText)
    print('answer:',answer.answerText)
    print('sim_arr:',sim_arr)
    if(sim_arr>0.5):
      score=score+1
    print('score:',score)
  finalScore = (score/(len(answers)))*100
  return score
