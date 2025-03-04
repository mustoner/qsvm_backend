from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def myfhelo():
    return{"hello", "world"}

@app.post("/item")
def create_item(name:str, price:float, itemid:int ):
    return{"name":name, "price":price, itemid:101}

@app.delete("/item/{item_id}")
def dete_item(item_id:int):
    return{"message": f"item {item_id} delteted succeffully"}
