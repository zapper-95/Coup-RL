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


torch, nn = try_import_torch()

class ActionMaskCentralizedCritic(TorchModelV2, nn.Module):
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
        self.internal_model = TorchFC(
            obs_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
            )

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 6 + 6 + 2  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]}, state, seq_lens)


        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        print(opponent_obs["observations"], opponent_actions, obs["observations"])
        input_ = torch.cat(
            [
                obs["observations"],
                opponent_obs["observations"],
                torch.nn.functional.one_hot(opponent_actions.long(), 2).float(),
            ],
            1,
        )
        return torch.reshape(self.central_vf(input_), [-1])
    
    @override(ModelV2)
    def value_function(self):
        return self.internal_model.value_function()