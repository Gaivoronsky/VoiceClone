import math
import traceback
from pathlib import Path

import torch
import torch.nn as nn

from models.VoiceClone import VoiceClone
from models.embedder import SpeechEmbedder
from utils.audio import Audio
from utils.adabound import AbaBound
from utils.evaluation import validate


def train(args, pt_dir, trainloader, testloader, writer, logger, hp, hp_str, device):
    embedder_pt = torch.load(args.embedder)
    embedder = SpeechEmbedder(hp).to(device)
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceClone(hp).to(device)
    if hp.train.optimizer == 'adabound':
        optimizer = AbaBound(
            model.parameters(),
            lr=hp.train.adabound.initial,
            final_lr=hp.train.adabound.final,
        )
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hp.train.adam,
        )
    else:
        raise Exception(f'{hp.train.optimizer} optimizer not supported')

    step = 0

    if args.checkpoint:
        logger.info(f'Resuming from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        if hp_str != checkpoint['hp_str']:
            logger.warning('New hparams is different from checkpoint')
    else:
        logger.info('Starting new training run')

    try:
        criterion = nn.MSELoss()
        while True:
            model.train()
            for dvec_mels, target_mag, other_mag in trainloader:
                target_mag, other_mag = target_mag.to(device), other_mag.to(device)

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.to(device)
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                mask = model(other_mag, dvec)
                output = other_mag * mask

                loss = criterion(output, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error('Loss exploded to %.02f at step %d!' % (loss, step))
                    raise Exception('Loss exploded')

                if step % hp.train.checkpoint_interval == 0:
                    save_path = Path(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info(f'Saved checkpoint to {save_path}')
                    validate(audio, model, embedder, testloader, writer, step, device)
    except Exception as e:
        logger.info(f'Exiting due to exception {e}')
        traceback.print_exc()
