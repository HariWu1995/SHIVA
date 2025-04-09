# SHIVA

1. **S**killset **H**ub with **I**ntelligence **V**irtual **A**cademy

2. **S**calable **H**ybrid **I**ntelligence in a **V**irtual **A**gency

3. **S**haring **H**ub and **I**ncubator with **V**irtual **A**gents

## Folder Structure

### SHIVA

    📦 SHIVA
    ├─ 📂 apps		# Submodules / local clones of AI apps
    │   ├─ 📂 repo_<service_name>
    │   ├─ 📂 …
    ├─ 📂 gateway	# Master API gateway
    │   ├─ 📄 config.py		# Service loading logic
    │   ├─ 📄 main.py		# Loads routers from `apps`
    │   └─ 📄 utils.py		# Dynamic import helpers
    ├─ 📂 uinified	# Unified Gradio interface
    │   ├─ 📄 config.py		# UI metadata config
    │   ├─ 📄 main.py		# Builds dynamic UI from `apps`
    │   └─ 📄 utils.py		# Dynamic import helpers
    ├─ 📂 configs   # Service discovery config
    │   ├─ 📄 apps.yaml		# [Optional] app config
    │   └─ 📄 services.yaml # Service Registry / Discovery
    ├─ 📂 controller
    │   ├─ 📄 orchestrator.py
    │   └─ 📄 utils.py
    ├─ 📄 docker-compose.yaml
    └─ …


### AI Application

    📦 repo-<service_name>
    ├─ 📂 src		# Core AI Logic - reuse original source code
    ├─ 📂 fapi		# FastAPI Application
    | ├─ 📄 api.py      # Inference routes
    | ├─ 📄 config.py   # Configs like model paths, constants
    | ├─ 📄 main.py     # FastAPI entrypoint
    | ├─ 📄 service.py  # Core logic for model dispatch + inference
    ├─ 📂 grui      # Gradio app for user interface
    | ├─ 📂 assets      # Images, stylesheets
    | ├─ 📄 ui.py       # Gradio interface setup
    | ├─ 📄 utils.py    # Helpers for UI rendering
    ├─ …
    ├─ 📄 Dockerfile
    ├─ 📄 README.md         # Documentation
    └─ 📄 requirements.txt  # Libraries to install

<details>

<summary>List of applications</summary>

- 

</details>



