from fastapi import FastAPI, HTTPException
import uvicorn


# curl -X POST -H "Content-Type: application/json" 'http://127.0.0.1:8000/items?item=apple'
# curl.exe -X POST -H "Content-Type: application/json" "http://127.0.0.1:8000/items?item=orange"
# curl.exe -X POST "http://127.0.0.1:8000/items?item=apple"

# curl.exe -X GET "http://127.0.0.1:8000/items?limit=3"


app = FastAPI()

items = ['apple', 'orange', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple' ]

@app.get('/')
def root():
    return {'hello': 'world'}


@app.post('/items')
def create_item(item: str):
    items.append(item)
    return items

@app.get('/items')
def list_items(limit: int = 10):
    return items[0:limit]


@app.get('/items/{item_id}')
def get_item(item_id: int) -> str:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail = 'item not found')
