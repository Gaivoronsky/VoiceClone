import argparse
from pathlib import Path
import logging
import time

import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.dataloader_vctk import VCTKDataset
from utils.hparams import HParam
from utils.train import train
from utils.writer import MyWriter


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        default='data',
        help='Training data (default: data)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Size batch train (default: 32)',
    )
    parser.add_argument(
        '--batch-size-test',
        type=int,
        default=1,
        help='Size batch valid (default: 1)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Value LR for train model (default: 0.0001)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number epochs to train (default: 10)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--size_valid',
        type=float,
        default=0.2,
        help='Number valid data of all data (default: 0.2)',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model for pre-train (default: None)',
    )
    parser.add_argument(
        '--embedded',
        type=str,
        default='embedder.pt',
        help='Path to model embedded (default: embedder.pt)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/default.yaml',
        help='Path to config file (default: ./config/default.yaml)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='VoiceClone',
        help='Name model (default: VoiceClone)',
    )
    return parser.parse_args()


def train_collate_fn(batch):
    dvec_list = list()
    target_mag_list = list()
    mixed_mag_list = list()

    for dvec_mel, target_mag, mixed_mag in batch:
        dvec_list.append(dvec_mel)
        target_mag_list.append(target_mag)
        mixed_mag_list.append(mixed_mag)
    target_mag_list = torch.stack(target_mag_list, dim=0)
    mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

    return dvec_list, target_mag_list, mixed_mag_list


def test_collate_fn(batch):
    return batch


def main(args):
    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    base_dir = Path(__file__).absolute().parent

    pt_dir = Path(base_dir, hp.log.chkpt_dir, args.model)
    pt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(base_dir, hp.log.log_dir, args.model)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir, '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(device)

    writer = MyWriter(hp, str(log_dir))

    datasets = torchaudio.datasets.VCTK_092('.', audio_ext='.wav', download=True)

    corpus = [sample for sample in tqdm(datasets)]
    df = pd.DataFrame(corpus, columns=['wav', 'sr', 'transcript', 'speaker_id', 'utterance_id'])

    train_data, test_data = train_test_split(df, test_size=0.1)
    train_data.sort_values(by=['utterance_id'], inplace=True)
    test_data.sort_values(by=['utterance_id'], inplace=True)

    trainloader = DataLoader(
        dataset=VCTKDataset(train_data, hp),
        batch_size=hp.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
        sampler=None,
    )
    testloader = DataLoader(
            dataset=VCTKDataset(test_data, hp, train=False),
            collate_fn=test_collate_fn,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=0,
        )

    train(args, pt_dir, trainloader, testloader, writer, logger, hp, hp_str, device)


if __name__ == '__main__':
    args = create_parser()
    main(args)
