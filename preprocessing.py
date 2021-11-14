import os
from pathlib import Path

import torchaudio
from utils.convert import ConvertData
from queue import Queue
from tqdm import tqdm


def main():
    torchaudio.datasets.VCTK_092('.', download=True)

    num_thread = os.cpu_count()
    print(f"{num_thread} cores available")

    dir_data = Path('VCTK-Corpus-0.92')
    list_path_audio = dir_data.glob('wav48_silence_trimmed/*/*.flac')

    queue = Queue()

    for _ in range(num_thread):
        thread = ConvertData(queue)
        thread.setDaemon(True)
        thread.start()

    for path_audio in tqdm(list_path_audio):
        queue.put(path_audio)

    print('Now wait about half an hour')
    queue.join()


if __name__ == '__main__':
    main()
