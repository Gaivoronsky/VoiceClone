import os
from pathlib import Path

import torchaudio
from utils.convert import ConvertData
from queue import Queue


def main():
    torchaudio.datasets.VCTK_092('.', download=True)

    num_thread = os.cpu_count()
    print(f"{num_thread} cores available")

    dir_data = Path('VCTK-Corpus-0.92')
    list_path_audio = list(dir_data.glob('wav48_silence_trimmed/*/*.flac'))
    size = len(list_path_audio) // num_thread
    list_thread = [list_path_audio[idx * size: (idx + 1) * size] for idx in range(len(list_path_audio) // size + 1)]

    queue = Queue()

    for _ in range(num_thread):
        thread = ConvertData(queue)
        thread.setDaemon(True)
        thread.start()

    for list_audio in list_thread:
        queue.put(list_audio)

    print('Now wait about half an hour')
    queue.join()


if __name__ == '__main__':
    main()
