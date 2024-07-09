import torch
import math


class SigmaifLayer(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, K: int, aggregation_threshold: float) -> None:
        super().__init__()
        
        # Linear
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty((out_features, in_features))
        self.weight = torch.nn.Parameter(weight)

        bias = torch.empty(out_features)
        self.bias = torch.nn.Parameter(bias)

        self._init_parameters()

        # Sigma-if
        self.K = K
        self.aggregation_threshold = aggregation_threshold
        
        self.in_groups = []
        self._init_in_groups()

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        '''
            input.size() = torch.Size([batch_size, block_size, in_features])
            output.size() = torch.Size([batch_size, block_size, out_features])
        '''
        size = input.size()
        input = input.view(-1, self.in_features) # torch.Size([batch_size * block_size, in_features])

        # TODO: To parallelize
        output = []
        for x in input:
            y = self._sigmaif_forward(x, self.weight, self.bias)
            output.append(y)
        output = torch.stack(output)

        output = output.view(size[0], size[1], self.out_features)
        return output

    def _sigmaif_forward(self, x: torch.FloatTensor, w: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
        start_index = 0
        partial_activation = 0
        for k in range(self.K):
            end_index = self.in_features - self.in_groups.index(k) # in_groups is reversed!

            partial_weight = w[:, start_index:end_index]
            partial_input = x[start_index:end_index]

            partial_Wx = partial_weight @ partial_input
            partial_EWx = torch.sum(partial_Wx)
            partial_activation = partial_EWx + partial_activation

            if partial_activation >= self.aggregation_threshold:
                break

            start_index = end_index
        return torch.sigmoid(partial_activation + b)

    def _init_in_groups(self) -> None:
        if self.K >= self.in_features:
            self.in_groups = [x for x in range(self.in_features)]
        else:
            quotient = self.in_features // self.K
            reminder = self.in_features % self.K

            self.in_groups = []
            for k in range(self.K):
                if k < self.K - 1:
                    g = [k] * quotient
                else:
                    g = [k] * (quotient + reminder)
                self.in_groups.extend(g)
        self.in_groups.reverse() # Reverse at this point is performed only once

    def _init_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)
