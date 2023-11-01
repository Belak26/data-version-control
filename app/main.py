import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
app = FastAPI()


class Iris(BaseModel):
    pl: float
    pw: float
    sl: float
    sw: float


with open("modelIris.pickle", "rb") as openfile:
    model1 = pickle.load(openfile)

@app.post("/api/v0/classify")
async def predict_iris(iris: Iris):
    pl = iris.pl
    pw = iris.pw
    sl = iris.sl
    sw = iris.sw

    iris_input = [pl, pw, sl, sw]

    prediction = model1.predict(iris_input)

    dict_aux = {0: "Setosa",
                1: "Versicolour",
                2: "Virginica"}

    return {"Iris Flower": dict_aux[prediction]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, log_level="info", reload=False)
