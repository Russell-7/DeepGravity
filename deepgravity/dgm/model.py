from torch import nn
import torch
class DeepGravity(nn.Module):
    def __init__(self, config):
        super(DeepGravity, self).__init__()
        self.config = config
        self.embed = nn.Embedding(num_embeddings=280,embedding_dim=128,dtype=None)
        self.linear_in = nn.Linear(9*128, 64)
        self.linears = nn.ModuleList(
            [nn.Linear(64, 64) for i in range(5)]
        )
        self.linear_out = nn.Linear(64, 1)

    def forward(self, input):
        input = self.embed(input).view(-1,9*128)
        input = self.linear_in(input)
        x = input
        for layer in self.linears:
            x = torch.relu(layer(x))
        x = x + input
        x = torch.sigmoid(self.linear_out(x))
        return x