import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import numpy
import sys

print(torch.__version__)
print(torchaudio.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
