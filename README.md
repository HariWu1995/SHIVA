# SHIVA

1. **S**killset **H**ub with **I**ntelligence **V**irtual **A**cademy

2. **S**calable **H**ybrid **I**ntelligence in a **V**irtual **A**gency

3. **S**haring **H**ub and **I**ncubator with **V**irtual **A**gents

## Folder Structure

### SHIVA

    📦 SHIVA
    ├─ 📂 apps		# Submodules / local clones of AI apps
    │   ├─ 📂 <service_name>
    │   └─ 📂 …
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

### AI Service

    📦 <service_name>
    ├─ 📂 src		# Core AI Logic - reuse original source code
    ├─ 📂 fapi		# FastAPI Application
    | ├─ 📄 api.py      # Inference routes
    | ├─ 📄 config.py   # Configs like model paths, constants
    | ├─ 📄 main.py     # FastAPI entrypoint
    | └─ 📄 service.py  # Core logic for model dispatch + inference
    ├─ 📂 grui      # Gradio app for user interface
    | ├─ 📂 assets      # Images, stylesheets
    | ├─ 📄 ui.py       # Gradio interface setup
    | └─ 📄 utils.py    # Helpers for UI rendering
    ├─ …
    ├─ 📄 Dockerfile
    ├─ 📄 README.md         # Documentation
    └─ 📄 requirements.txt  # Libraries to install

## Application

### Preprocessing

- [x] Background Removal ([RemBg](https://github.com/HariWu1995/Anilluminus.AI/tree/main/src/apps/rembg))

- [ ] Controlnet Preprocessors ([ControlNet](https://github.com/Mikubill/sd-webui-controlnet))

- [ ] Object Segmentation ([SAM-WebUI](https://github.com/5663015/segment_anything_webui))

- [ ] Pose Keypoint ([JavaScript](https://github.com/Mikubill/sd-webui-controlnet/blob/main/javascript/openpose_editor.mjs))

- [ ] Camera Trajectory ([Viser](https://github.com/Stability-AI/stable-virtual-camera/blob/main/demo_gr.py#L769))

### Uni-modal Generation

- [ ] Text Generation ([WebUI](https://github.com/oobabooga/text-generation-webui))

- [ ] Image Generation ([AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

### View Synthesis

- [ ] Text-to-MV ([MVDream](https://github.com/bytedance/MVDream) | [MVDiffusion](https://github.com/Tangshitao/MVDiffusion))

- [ ] Image-to-MV ([Zero123++](https://github.com/SUDO-AI-3D/zero123plus) | [MVDiffusion](https://github.com/Tangshitao/MVDiffusion))

- [ ] Image/Text-to-Panorama ([SD-T2I-360PanoImage](https://github.com/ArcherFMY/SD-T2I-360PanoImage))

- [ ] Camera Control ([Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera) | [CameraCtrl](https://github.com/hehao13/CameraCtrl))

<details>
    <summary><i>Notation</i></summary>

- <b>MV</b>: Multi-view

</details>

### Editing

- [ ] Image Editing ([MagicQuill](https://github.com/ant-research/MagicQuill))

- [ ] Re-Lighting ([ICLight](https://github.com/lllyasviel/IC-Light))

- [ ] Style Transfer ([CSGO](https://github.com/instantX-research/CSGO))

### 3D Reconstruction

- [ ] Image-to-3D-Anime-Character ([CharacterGen](https://github.com/zjp-shadow/CharacterGen))

- [ ] Image-to-3D ([Unique3D](https://github.com/AiuniAI/Unique3D) | [StableFast3D](https://github.com/Stability-AI/stable-fast-3d) | [InstantMesh](https://github.com/TencentARC/InstantMesh))

- [ ] Multi-Instance-to-3D ([MIDI-3D](https://github.com/VAST-AI-Research/MIDI-3D))

### Animation

- [ ] 2D Animation ([Animate Anything](https://github.com/alibaba/animate-anything) | [MOFA-Video](https://github.com/MyNiuuu/MOFA-Video))

- [ ] 3D Animation ([Make It Animatable](https://github.com/jasongzy/Make-It-Animatable))

### Virtual Try-on (ViTon)

- [ ] Multi-View ViTon ([MV-VTON](https://github.com/hywang2002/MV-VTON))

- [ ] Full-Body ViTon ([OOTDiffusion](https://huggingface.co/spaces/levihsu/OOTDiffusion) | [PICTURE](https://github.com/GAP-LAB-CUHK-SZ/PICTURE))


