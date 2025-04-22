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
