import argparse
import librosa
import torch
from tensorboard.plugins.hparams.hparams_util_pb2 import HParams
import soundfile as sf

from models.VoiceClone import VoiceClone
from models.embedder import SpeechEmbedder
from utils.audio import Audio


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        default='./config/default.yaml',
        help="Path to config file (default: ./config/default.yaml)",
    )
    parser.add_argument(
        '-e', '--embedder',
        default='./embedder.pt',
        help='Path to model embedded (default: embedder.pt)',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help="Path of checkpoint pt file",
    )
    parser.add_argument(
        '-a', '--audio',
        required=True,
        help='Path to audio for processing',
    )
    parser.add_argument(
        '-r', '--reference',
        required=True,
        help='Path to audio with voice target'
    )
    parser.add_argument(
        '-o', '--output',
        default='./voice.wav',
        help='Path for save predicted audio (default: ./voice.wav)'
    )
    return parser.parse_args()


def main(args):
    hp = HParams(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    with torch.no_grad():
        model = VoiceClone(hp).to(device)
        chkpt_model = torch.load(args.checkpoint)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).to(device)
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)
        dvec_wav, _ = librosa.load(args.reference, sr=16000)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().to(device)
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        other_wav, _ = librosa.load(args.audio, sr=16000)
        mag, phase = audio.wav2spec(other_wav)
        mag = torch.from_numpy(mag).float().to(device)

        mag = mag.unsqueeze(0)
        mask = model(mag, dvec)
        est_mag = mag * mask

        est_mag = est_mag[0].cpu().detach().numpy()
        est_wav = audio.spec2wav(est_mag, phase)

        sf.write(args.output, est_wav, 16000)


if __name__ == '__main__':
    args = create_parser()
    main(args)
