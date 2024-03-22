from ray.tune.logger import pretty_print
import coup_v1
import torch
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from gymnasium.spaces import MultiDiscrete, Discrete
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.rllib.algorithms.dqn import DQNConfig


class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: MultiDiscrete,
        action_space: Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )


        self.action_embed_model = TorchFC(
            obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()



# coupenv = coup_v1.env()


ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)

def env_creator():
    env = coup_v1.env()
    return env


env_name = "coup_v1"

register_env(env_name, lambda config: PettingZooEnv(env_creator()))


test_env = PettingZooEnv(env_creator())
obs_space = test_env.observation_space
act_space = test_env.action_space

config = (DQNConfig()
            .environment(env=env_name)
            .rollouts(num_rollout_workers=2, create_env_on_local_worker=True)
            .training(
                train_batch_size=200,
                hiddens=[],
                dueling=False,
                model={"custom_model": "pa_model"},
            )
            .multi_agent(
                policies={
                    "player_1": (None, obs_space, act_space, {}),
                    "player_2": (None, obs_space, act_space, {}),
                },
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            )
          
          )


pretty_print(config.to_dict())

algo = config.build()

for i in range(10):
    result = algo.train()

print(pretty_print(result))