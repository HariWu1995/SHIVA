# SHIVA

1. **S**killset **H**ub with **I**ntelligence **V**irtual **A**cademy

2. **S**calable **H**ybrid **I**ntelligence in a **V**irtual **A**gency

3. **S**haring **H**ub and **I**ncubator with **V**irtual **A**gents

## Folder Structure

### SHIVA

    📦 SHIVA
    ├─ 📂 apps       # Submodules / local clones of AI apps
    │   ├─ 📂 <service_group>
    │   │   ├─ 📂 <service_name>
    │   │   └─ 📂 …
    │   └─ 📂 …
    ├─ 📂 gateway    # Master API gateway
    │   ├─ 📄 config.py       # Service loading logic
    │   ├─ 📄 main.py         # Loads routers from `apps`
    │   └─ 📄 utils.py        # Dynamic import helpers
    ├─ 📂 uinified   # Unified Gradio interface
    │   ├─ 📄 config.py       # UI metadata config
    │   ├─ 📄 main.py         # Builds dynamic UI from `apps`
    │   └─ 📄 utils.py        # Dynamic import helpers
    ├─ 📂 configs    # Service discovery config
    │   ├─ 📄 apps.yaml       # [Optional] app config
    │   └─ 📄 services.yaml   # Service Registry / Discovery
    ├─ 📂 controller
    │   ├─ 📄 orchestrator.py
    │   └─ 📄 utils.py
    ├─ 📄 docker-compose.yaml
    └─ …

### AI Service

    📦 <service_name>
    ├─ 📂 src        # Core AI Logic - reuse original source code
    ├─ 📂 fapi        # FastAPI Application
    | ├─ 📄 api.py      # Inference routes
    | ├─ 📄 config.py   # Configs like model paths, constants
    | ├─ 📄 main.py     # FastAPI entrypoint
    | └─ 📄 service.py  # Core logic for model dispatch + inference
    ├─ 📂 grui      # Gradio app for user interface
    | ├─ 📂 assets      # Images, stylesheets
    | ├─ 📄 main.py     # Gradio interface layout
    | └─ 📄 utils.py    # Helpers for UI rendering
    ├─ …
    ├─ 📄 Dockerfile
    ├─ 📄 README.md         # Documentation
    └─ 📄 requirements.txt  # Libraries to install

## Services

### Preprocessing

- [x] Background Decomposition ([RemBg](https://github.com/HariWu1995/Anilluminus.AI/tree/main/src/apps/rembg))

- [x] Instance Segmentation ([OWL-ViT+SAM](https://huggingface.co/spaces/SkalskiP/florence-sam) | [Florence+SAM](https://huggingface.co/spaces/SkalskiP/florence-sam))

- [ ] 3D Object Segmentation ([Point-SAM](https://github.com/zyc00/Point-SAM))

- [x] Video Matting ([RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting))

- [x] Controlnet Preprocessors ([ControlNet](https://github.com/Mikubill/sd-webui-controlnet))

- [x] OpenPose Editor ([2D Editor](https://github.com/huchenlei/sd-webui-openpose-editor) | [3D Editor](https://github.com/ZhUyU1997/open-pose-editor/releases))

### Uni-modal Generation

- [ ] Text Generation ([WebUI](https://github.com/oobabooga/text-generation-webui))

- [ ] Image Generation ([AUTOMATIC-1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui))

- [ ] Audio Generation ([Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio) | [AudioLDM](https://github.com/haoheliu/AudioLDM) | [Amphion](https://github.com/open-mmlab/Amphion) | [AudioCraft](https://github.com/facebookresearch/audiocraft) | [Suno|Bark](https://github.com/suno-ai/bark))

### View Synthesis

- [ ] Text-to-MV ([MVDream](https://github.com/bytedance/MVDream) | [MVDiffusion](https://github.com/Tangshitao/MVDiffusion))

- [ ] Image-to-MV ([Zero123++](https://github.com/SUDO-AI-3D/zero123plus) | [MVDiffusion](https://github.com/Tangshitao/MVDiffusion))

- [ ] Image/Text-to-Panorama ([SD-T2I-360PanoImage](https://github.com/ArcherFMY/SD-T2I-360PanoImage))

- [ ] Camera Control ([SeVa](https://github.com/Stability-AI/stable-virtual-camera) | [CameraCtrl](https://github.com/hehao13/CameraCtrl))

- [x] 3D Visualization ([Viser](https://github.com/nerfstudio-project/viser) | [Camera Trajectory|SeVa](https://github.com/Stability-AI/stable-virtual-camera/blob/main/demo_gr.py#L769))

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

- [ ] Face Animation ([X-Pose](https://github.com/IDEA-Research/X-Pose) | [MimicTalk](https://github.com/yerfor/MimicTalk/) | [GeneFace++](https://github.com/yerfor/GeneFacePlusPlus/) | [Real3DPortrait](https://github.com/yerfor/Real3DPortrait))

### Virtual Try-on (ViTon)

- [ ] Multi-View ViTon ([MV-VTON](https://github.com/hywang2002/MV-VTON))

- [ ] Full-Body ViTon ([OOTDiffusion](https://huggingface.co/spaces/levihsu/OOTDiffusion) | [PICTURE](https://github.com/GAP-LAB-CUHK-SZ/PICTURE))

### Sound

- [ ] Music Generation ([MusicGPT](https://github.com/gabotechs/MusicGPT))

- [ ] Voice Clone ([Fish Speech](https://github.com/fishaudio/fish-speech) | [OpenVoice](https://github.com/myshell-ai/OpenVoice) | [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) | [Bark-Voice-Clone](https://github.com/serp-ai/bark-with-voice-clone))

- [ ] Voice Clone VN ([viXTTS](https://github.com/thinhlpg/vixtts-demo) | [EraX-Female](https://huggingface.co/erax-ai/EraX-Smile-Female-F5-V1.0))


## Applications

- [ ] Digital Clone ([HeyGem](https://github.com/GuijiAI/HeyGem.ai))


