import os
from pathlib import Path
from tqdm import tqdm
from tinytag import TinyTag
from threading import Thread


class ConvertData(Thread):
    def __init__(self, list_audio: list):
        Thread.__init__(self)
        self.list_audio = list_audio

    def _len_audio(self, fname: Path) -> float:
        tag = TinyTag.get(fname)
        return tag.duration

    def convert_dataset(self):
        for path_wav in tqdm(self.list_audio):
            final_path_wav = str(path_wav).replace('flac', 'wav')
            if self._len_audio(path_wav) > 3:
                os.system(f'ffmpeg-normalize {path_wav} -ar 16000 -o {final_path_wav}')
            else:
                try:
                    path_txt = str(final_path_wav).replace('wav48_silence_trimmed', 'txt')
                    path_txt = path_txt[:-9] + '.txt'
                    os.remove(path_txt)
                except:
                    pass
