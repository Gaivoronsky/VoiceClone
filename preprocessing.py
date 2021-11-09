import torchaudio
from utils.convert import convert_dataset


def main():
    torchaudio.datasets.VCTK_092('.', download=True)
    convert_dataset('VCTK-Corpus-0.92')


if __name__ == '__main__':
    main()
