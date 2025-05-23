"""
Robot URDF visualizer
    https://github.com/nerfstudio-project/viser/blob/main/examples/09_urdf_visualizer.py

Requires yourdfpy and robot_descriptions. Any URDF supported by yourdfpy should work.
    - https://github.com/robot-descriptions/robot_descriptions.py
    - https://github.com/clemense/yourdfpy

The :class:`viser.extras.ViserUrdf` is a lightweight interface between yourdfpy
and viser. It can also take a path to a local URDF file as input.
"""
from __future__ import annotations
from typing import Literal, get_args

import time
import numpy as np

import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description


ROBOT_TYPES = Literal["panda","ur10","cassie","allegro_hand","barrett_hand",
                      "robotiq_2f85","atlas_drc","g1","h1","anymal_c","go2"]
DEFAULT_TYPE = "panda"
ALLOWED_TYPES = list(get_args(ROBOT_TYPES))


def create_robot_control_sliders(
    server: viser.ViserServer, 
    viser_urdf: ViserUrdf,
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """
    Create slider for each joint of the robot. We also update robot model when slider moves.
    """
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []

    for joint_name, (lower, upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(np.array([slider.value for slider in slider_handles]))
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main(
    robot_type: ROBOT_TYPES,
    host: str = "0.0.0.0",
    port: int = 8080,
    share: bool = False,
) -> None:
    # Start viser server.
    server = viser.ViserServer(host=host, port=port)
    if share:
        server.request_share_url()

    # Load URDF.
    # This takes either a yourdfpy.URDF object or a path to a .urdf file.
    viser_urdf = ViserUrdf(server,
                urdf_or_path=load_robot_description(robot_type + "_description"))

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(server, viser_urdf)

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create grid.
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(0.0, 0.0, viser_urdf._urdf.scene.bounds[0, 2]) # min z-value of trimesh scene.
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
