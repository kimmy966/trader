import torch
from torch.autograd import Variable
import numpy as np


class DQN(torch.nn.Module):
    def __init__(self, num_state, num_action, num_hiddens):
        super().__init__()
        self.num_state = num_state
        self.num_action = num_action

        self.linear1 = torch.nn.Linear(num_state, num_hiddens)
        self.activation1 = torch.tanh
        self.linear2 = torch.nn.Linear(num_hiddens, num_action)
        # self.linear_out = torch.nn.Linear(num_hiddens, num_action)

    def forward(self, state, grad=True):
        assert isinstance(state, (np.ndarray, torch.FloatTensor)), state
        input = torch.FloatTensor(state) if isinstance(state,
            np.ndarray) else state
        x = Variable(input.view(-1, self.num_state), requires_grad=grad)
        out1 = self.linear1(x)
        out2 = self.activation1(out1)
        out = self.linear2(out2)
        # out = self.linear_out(out)
        out = out.view(list(input.size())[:-1] + [self.num_action])
        return out

    def sync(self, other):
        for dest, src in zip(self.parameters(), other.parameters()):
            dest.data.copy_(src.data)
