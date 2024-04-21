from ray.rllib.examples.models.centralized_critic_models import TorchCentralizedCriticModel
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from gymnasium.spaces import Dict
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
import coup_v2
from ray.rllib.env import PettingZooEnv


torch, nn = try_import_torch()

class ActionMaskCentralisedCritic(TorchModelV2, nn.Module):
    """Example of a model that implements both action masking and a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        # TorchActionMaskModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        # TorchCentralizedCriticModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # Base of the model
        self.model = TorchFC(
            obs_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name,
            )
        

        input_size = len(coup_v2.env().observation_space("player_1")["observations"])

        self.central_vf = nn.Sequential(
            SlimFC(input_size, 128, activation_fn=nn.Tanh),
            SlimFC(128, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.model({"obs": input_dict["obs"]["observations"]}, state, seq_lens)


        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state



    def adjust_tensor_size(self, tensor, target_size):
        """Adjust the size of the observation and action tensors to be as large as the reward tensor."""
        current_size = tensor.size(0)
        if current_size < target_size:
            
            # if it is the action tensor, add a pass action to it
            if len(tensor.size()) == 1:
                last_row = torch.tensor([9])
            else:
                # extend observation tensors by repeating the last row
                last_row = tensor[-1].unsqueeze(0).repeat(target_size - current_size, 1)
            adjusted_tensor = torch.cat((tensor, last_row), dim=0)
        elif current_size > target_size:
            # Truncate the tensor
            adjusted_tensor = tensor[:target_size]
        else:
            adjusted_tensor = tensor
        return adjusted_tensor

    def central_value_function(self, rewards, obs, opponent_obs):
        if type(obs) is dict:
            obs = obs["observations"]
        
        if type(opponent_obs) is dict:
            opponent_obs = opponent_obs["observations"]

        # take only the relevant observations, which are the hidden cards, and discard the rest
        opponent_obs = opponent_obs[:, 0:2]

        # remove partial observations about the opponent, from the normal observations
        obs = torch.cat((obs[:, :5], obs[:, 7:]), dim=1)

    
        expected_size = rewards.size(0)


        obs = self.adjust_tensor_size(obs, expected_size)

        opponent_obs = self.adjust_tensor_size(opponent_obs, expected_size)


        input_ = torch.cat(
            [
                obs,
                opponent_obs.float(),
            ],
            1,
        )
        return torch.reshape(self.central_vf(input_), [-1])
    
    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()