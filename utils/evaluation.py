import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources

from utils.audio import Audio
from utils.writer import MyWriter
from torch.utils.data import DataLoader


def validate(audio: Audio,
             model: nn.Module,
             embedder: nn.Module,
             testloader: DataLoader,
             writer: MyWriter,
             step: int,
             device: str):
    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in testloader:
            dvec_mel, target_wav, other_wav, target_mag, other_mag, other_phase = batch[0]

            dvec_mel = dvec_mel.to(device)
            target_mag = target_mag.unsqueeze(0).to(device)
            other_mag = other_mag.unsqueeze(0).to(device)

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            est_mask = model(other_mag, dvec)
            est_mag = est_mask * other_mag
            test_loss = criterion(target_mag, est_mag).item()

            other_mag = other_mag[0].cpu().detach().numpy()
            target_mag = target_mag[0].cpu().detach().numpy()
            est_mag = est_mag[0].cpu().detach().numpy()
            est_wav = audio.spec2wav(est_mag, other_phase)
            est_mask = est_mask[0].cpu().detach().numpy()

            sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            writer.log_evaluation(test_loss, sdr,
                                  other_wav, target_wav, est_wav,
                                  other_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                  step)
            break

    model.train()
