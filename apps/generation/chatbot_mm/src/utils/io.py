import os
from pathlib import Path, PurePath
from ..logging import logger


def save_file(fpath: str | PurePath, contents):
    if fpath == '':
        logger.error('File path is empty!')
        return

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(contents)
    logger.info(f'Saved \"{fpath}\".')


def delete_file(fpath: str | PurePath):
    if fpath == '':
        logger.error('File path is empty!')
        return

    if isinstance(fpath, str):
        fpath = Path(fpath)

    if fpath.exists():
        fpath.unlink()
        logger.info(f'Deleted \"{fpath}\".')


image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}
audio_extensions = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'}
video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}

def get_file_type(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext == '':
        return 'null'
    elif ext in image_extensions:
        return 'image'
    elif ext in audio_extensions:
        return 'audio'
    elif ext in video_extensions:
        return 'video'
    else:
        return 'unknown'


if __name__ == "__main__":
    print(get_file_type("C:/Users/Mr. RIAH/Pictures/event_layout.jpg"))
    print(get_file_type("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"))

