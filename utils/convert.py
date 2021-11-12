import os
from pathlib import Path
from tqdm import tqdm
from tinytag import TinyTag


def len_audio(fname: Path) -> float:
    tag = TinyTag.get(fname)
    return tag.duration


def convert_dataset(dir_data: str):
    dir_data = Path(dir_data)
    list_path_audio = dir_data.glob('wav48_silence_trimmed/*/*.flac')

    for path_wav in tqdm(list_path_audio):
        final_path_wav = str(path_wav).replace('flac', 'wav')
        if len_audio(path_wav) > 3:
            os.system(f'ffmpeg-normalize {path_wav} -ar 16000 -o {final_path_wav}')
        else:
            try:
                path_txt = str(final_path_wav).replace('wav48_silence_trimmed', 'txt')
                path_txt = path_txt[:-9] + '.txt'
                os.remove(path_txt)
            except:
                pass
