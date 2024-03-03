from fastapi import FastAPI

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json

from agent import process_input

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/chat", response_class=HTMLResponse)
async def chat_index(request: Request):
    return templates.TemplateResponse(
        request=request, name="chat.html", context={"id": ""}
    )


@app.post("/chat")
async def process_chat_input(request: Request):
    # Process the user input
    body = await request.body()
    user_message = json.loads(body.decode("utf-8"))

    response_message = process_input(user_message["user_message"])
    return {"response_message": response_message}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
