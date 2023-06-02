from fvcore.nn import FlopCountAnalysis
import torch
from torch import nn

class WrapperModule(nn.Module):

    def __init__(self, wrapped_model):
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(self, inputs):
        return self.wrapped_model(**inputs)


def FlopCountingTrainer(_Trainer):

    class _FlopCountingTrainer(_Trainer):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._flop_count = 0
            self._flop_counting_enabled = False

        def prediction_step(self, model, inputs, *args, **kwargs):
            #print('Computing loss!')
            if self._flop_counting_enabled:
                with torch.no_grad():
                    wrapped_model = WrapperModule(model)
                    prepared_inputs = self._prepare_inputs(inputs)
                    flops = FlopCountAnalysis(wrapped_model, prepared_inputs)
                    self._flop_count += flops.total()
            return super().prediction_step(model, inputs, *args, **kwargs)

        def evaluate(self, *args, **kwargs):
            self._flop_counting_enabled = True
            self._flop_count = 0
            metrics = super().evaluate(*args, **kwargs)
            self._flop_counting_enabled = False
            metrics['eval_flops'] = self._flop_count
            return metrics

    return _FlopCountingTrainer
