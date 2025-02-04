import torch.nn as nn

class RouterNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
        
    def forward(self, hidden_states):
        return self.net(hidden_states[:, -1])  # Use last token's hidden state
