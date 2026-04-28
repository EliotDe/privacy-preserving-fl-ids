"""
This class lets the client set DP parameters instead of having them pre-defined. 

NOTE: In the past I have received some divide-by-zero errors. The parent LocalDpMod
      class throws this error while calculating the clipping norm if the gradient 
      doesn't update during local training. This can occur when model parameters 
      explode when training with lots of noise for few rounds. I haven't experienced
      this in the final experiments but it's something to be aware of. 
"""

from flwr.app import Message, Context
from flwr.clientapp.mod import LocalDpMod
from flwr.clientapp.typing import ClientAppCallable


class DynamicDpMod(LocalDpMod):
    def __init__(self):
        """
        Initialize with default values
        """
        super().__init__(
                clipping_norm = 1.0,
                sensitivity = 1.0,
                epsilon=1.0,
                delta = 0.0001
        )

    def __call__(
            self, msg: Message, context: Context, call_next: ClientAppCallable
    ) -> Message:
        """
        Override default dp parameters to use config settings
        """
        eps = context.run_config["epsilon"]
        delta = context.run_config["delta"]
        sensitivity = context.run_config["sensitivity"]
        clipping_norm = context.run_config["clipping_norm"]
        self.epsilon = float(eps)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.clipping_norm = float(clipping_norm)


        return super().__call__(msg, context, call_next)