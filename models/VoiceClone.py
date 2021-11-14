import torch.nn as nn
import torch.nn.functional as F
import torch


class VoiceClone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        assert self.hp.audio.n_fft // 2 + 1 == self.hp.audio.num_freq == self.hp.model.fc2_dim, \
            "stft-related dimension mismatch"

        self.conv = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.Conv2d(64, 8, kernel_size=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            8 * self.hp.audio.num_freq + self.hp.embedder.emb_dim,
            self.hp.model.lstm_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(2 * self.hp.model.lstm_dim, self.hp.model.fc1_dim)
        self.fc2 = nn.Linear(self.hp.model.fc1_dim, self.hp.model.fc2_dim)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        # dvec: [B, T, emb_dim]
        dvec = dvec.repeat(1, x.size(1), 1)

        # [B, T, 8*num_freq + emb_dim]
        x = torch.cat((x, dvec), dim=2)

        # [B, T, 2*lstm_dim]
        x, _ = self.lstm(x)
        x = F.relu(x)
        # x: [B, T, fc1_dim]
        x = self.fc1(x)
        x = F.relu(x)
        # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
