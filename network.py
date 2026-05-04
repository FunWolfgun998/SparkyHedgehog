# network.py (ex newwork.py)
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Shally(BaseFeaturesExtractor):
    """
    SHALLY: Il Cervello (Rete Neurale) del progetto SparkyHedgehog.
    Shally ha il compito di analizzare i frame del gioco e tradurli in concetti logici.
    """

    def __init__(self, observation_space, features_dim=512):
        super(Shally, self).__init__(observation_space, features_dim)

        # n_input_channels = 4 (Riceviamo 4 frame sovrapposti per percepire la velocità)
        n_input_channels = observation_space.shape[0]

        # GLI OCCHI DI SHALLY (Pipeline Visiva)
        self.cnn = nn.Sequential(
            # 1° STRATO: Visione grossolana (Bordi e Muri)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # 2° STRATO: Forme complesse (Angoli, curve)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # 3° STRATO: Dettagli minuscoli (Anelli, nemici)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # FLATTEN: Trasforma il quadrato in una stringa di pensieri
            nn.Flatten(),
        )

        # CALCOLO AUTOMATICO DELLE DIMENSIONI
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # IL PENSIERO ASTRATTO DI SHALLY
        # Trasforma i dati visivi in 512 neuroni decisionali
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        """Come l'informazione fluisce dentro Shally"""
        return self.linear(self.cnn(observations))