#!/usr/bin/env python3

# TODO -> self evaluation head
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel


class SelfEvalHead(nn.Module):

    def __init__(self, eval_strategy: str):
        self.fc = self.initialize_fc(eval_strategy)


    def forward(self, inputs):
        return self.fc(inputs)


    def initialize_fc(strategy: str, **params):
        match strategy:
            case "direct":
                return nn.Linear(params["dim"], 5)
            case "binned":
                return nn.Linear(params["dim"], 1)

class SelfEvaluation(nn.Module):

    def __init__(self,
                 base_model: PreTrainedModel,
                 eval_strategy: str):
        self.lm = base_model
        self.eval_head = SelfEvalHead(eval_strategy)


    def forward(self, inputs):
        pass
