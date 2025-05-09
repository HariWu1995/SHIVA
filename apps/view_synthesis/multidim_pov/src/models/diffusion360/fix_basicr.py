import os
from pathlib import Path


def fix_torchvision_version(venv_path: str):

    # Path to the file you want to modify
    file_path = str(Path(venv_path).resolve() / 'Lib/site-packages/basicsr/data/degradations.py')

    # Old and new import lines
    old_line = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
    new_line = 'from torchvision.transforms.functional import rgb_to_grayscale'

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the specific line
    updated_lines = [
        new_line if old_line in line else line for line in lines
    ]

    # Write the updated content back
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Line updated in: {file_path}")


if __name__ == "__main__":

    path_to_venv = "C:/Users/Mr. RIAH/Documents/Projects/SHIVA/venv"
    fix_torchvision_version(path_to_venv)

