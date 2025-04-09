import subprocess
import json
import os
from typing import List, Dict, Optional
import shutil


class GPUInfo:

    def __init__(self, id: int, memory_total: int, memory_used: int):
        self.id = id
        self.memory_total = memory_total
        self.memory_used = memory_used

    def available_memory(self) -> int:
        return self.memory_total - self.memory_used


class Orchestrator:

    def __init__(self):
        self.gpu_list = self.detect_gpus()
        self.registry = {}

    def detect_gpus(self) -> List[GPUInfo]:
        """Detect GPUs and their memory status using nvidia-smi."""
        gpus = []
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi", 
                        "--query-gpu=index,memory.total,memory.used",
                        "--format=csv,noheader,nounits"
                    ], stdout=subprocess.PIPE, check=True, text=True
                )
                for line in result.stdout.strip().split("\n"):
                    idx, mem_total, mem_used = map(int, line.split(", "))
                    gpus.append(GPUInfo(id=idx, memory_total=mem_total, memory_used=mem_used))
            except subprocess.CalledProcessError as e:
                print(f"nvidia-smi error: {e}")
        return gpus

    def get_available_gpu(self) -> Optional[GPUInfo]:
        """Get the GPU with the most available memory."""
        if not self.gpu_list:
            return None
        return max(self.gpu_list, key=lambda gpu: gpu.available_memory())

    def start_service(self, app_name: str, image: str, ports: Dict[int, int], model_version: str) -> str:
        """Run a Docker container for the specified app on a suitable GPU."""
        gpu = self.get_available_gpu()
        if not gpu:
            raise RuntimeError("No available GPUs found.")

        container_name = f"{app_name}-{model_version}".replace("_", "-")
        port_args = " ".join([f"-p {host}:{container}" for host, container in ports.items()])

        cmd = (
            f"docker run -d --gpus 'device={gpu.id}' "
            f"--name {container_name} {port_args} "
            f"-e MODEL_VERSION={model_version} "
            f"{image}"
        )

        try:
            result = subprocess.check_output(cmd, shell=True, text=True).strip()
            self.registry[container_name] = {
                "gpu_id": gpu.id,
                "model_version": model_version,
                "container_id": result
            }
            return result
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start container: {e}")

    def stop_service(self, container_name: str) -> None:
        """Stop and remove a Docker container."""
        try:
            subprocess.run(f"docker rm -f {container_name}", shell=True, check=True)
            self.registry.pop(container_name, None)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to stop container {container_name}: {e}")

    def list_services(self) -> Dict[str, Dict]:
        """Return currently tracked services."""
        return self.registry

    def scale_services(self, usage_data: Dict[str, int], threshold: int = 10):
        """Example: scale up if requests > threshold, otherwise stop service."""
        for app_name, req_count in usage_data.items():
            container_name = next((c for c in self.registry if app_name in c), None)
            if req_count > threshold and container_name is None:
                self.start_service(app_name, f"{app_name}:latest", {8000: 80}, "v1")
            elif req_count == 0 and container_name:
                self.stop_service(container_name)

        # Refresh GPU info
        self.gpu_list = self.detect_gpus()

        return self.list_services()


if __name__ == "__main__":

    orch = Orchestrator()

    # Start a service on available GPU
    orch.start_service(
        app_name="image_classifier",
        image="image_classifier:latest",
        ports={8001: 80},
        model_version="v1"
    )

    # Stop a service
    orch.stop_service("image-classifier-v1")

    # Get running services
    orch.list_services()

    # Example auto-scaling based on usage
    orch.scale_services(usage_data={
        "image_classifier": 15, 
        "sentiment_analysis": 0,
    })

