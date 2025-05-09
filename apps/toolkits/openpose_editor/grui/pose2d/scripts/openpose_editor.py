import os
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class Item(BaseModel):
    # image url.
    image_url: str
    # stringified pose JSON.
    pose: str


CURR_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = os.path.join(CURR_DIR, 'dist')


def mount_openpose_api(app: FastAPI):

    templates = Jinja2Templates(directory=DIST_DIR)
    app.mount(
        "/openpose_editor",
        StaticFiles(directory=DIST_DIR, html=True),
        name="openpose_editor",
    )

    @app.get("/openpose_editor_index", response_class=HTMLResponse)
    async def index_get(request: Request):
        return templates.TemplateResponse(
            "index.html", {"request": request, "data": {}}
        )

    @app.post("/openpose_editor_index", response_class=HTMLResponse)
    async def index_post(request: Request, item: Item):
        return templates.TemplateResponse(
            "index.html", {"request": request, "data": item.dict()}
        )


if __name__ == "__main__":
    app = FastAPI()
    mount_openpose_api(app)

    import uvicorn
    uvicorn.run(app, host="localhost", port=5000, reload=False)
