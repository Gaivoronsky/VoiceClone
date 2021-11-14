import os
from pathlib import Path

import torchaudio
from utils.convert import ConvertData


def main():
    torchaudio.datasets.VCTK_092('.', download=True)

    num_thread = os.cpu_count()
    dir_data = Path('VCTK-Corpus-0.92')
    list_path_audio = list(dir_data.glob('wav48_silence_trimmed/*/*.flac'))
    size = len(list_path_audio) // num_thread
    list_thread = [list_path_audio[idx * size: (idx + 1) * size] for idx in range(len(list_path_audio) // size + 1)]

    for idx, list_audio in enumerate(list_thread):
        thread = ConvertData(list_audio)
        thread.start()


if __name__ == '__main__':
    main()
