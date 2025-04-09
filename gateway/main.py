from fastapi import FastAPI
import importlib
import os

app = FastAPI(title="Master Inference Gateway")

APPS_DIR = "apps"

# Discover and mount sub-routers
for app_name in os.listdir(APPS_DIR):
    app_path = f"{APPS_DIR}.{app_name}.app_api"
    try:
        module = importlib.import_module(app_path)
        router = getattr(module, "router")
        app.include_router(router, prefix=f"/inference/{app_name}")
        print(f"Mounted API from {app_name}")
    except (ImportError, AttributeError) as e:
        print(f"Skipping {app_name}: {e}")
