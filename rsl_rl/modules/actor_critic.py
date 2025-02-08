#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


# class TransformedActor(nn.Module):
#     def __init__(self, actor):
#         super().__init__()
#         self.actor = actor

    
#         # hip = _0
#         # thigh = _1
#         # calf = _2
            
            
#         self._hardware_joint_order = [
#             "FR_0", "FR_1", "FR_2", 
#             "FL_0", "FL_1", "FL_2",
#             "RR_0", "RR_1", "RR_2",
#             "RL_0", "RL_1", "RL_2"
#         ]

#         self._sim_joint_order = [
#             'FL_0', 'FR_0', 'RL_0',
#             'RR_0', 'FL_1', 'FR_1',
#             'RL_1', 'RR_1', 'FL_2',
#             'FR_2', 'RL_2', 'RR_2'
#         ]

#         self._hardware_to_sim_joint_order = [self._sim_joint_order.index(joint) for joint in self._hardware_joint_order]
#         self._sim_to_hardware_joint_order = [self._hardware_joint_order.index(joint) for joint in self._sim_joint_order]

#         self._joint_pos_idx = 3 + 3 + 3
#         self._joint_vel_idx = self._joint_pos_idx + 12
#         self._last_action_idx = self._joint_vel_idx + 12

#     def transform_observation(self, obs):
#         obs = obs.clone()
#         obs[:, self._joint_pos_idx: self._joint_pos_idx + 12] = obs[:, self._joint_pos_idx: self._joint_pos_idx + 12][:, self._sim_to_hardware_joint_order]
#         obs[:, self._joint_vel_idx: self._joint_vel_idx + 12] = obs[:, self._joint_vel_idx: self._joint_vel_idx + 12][:, self._sim_to_hardware_joint_order]
#         obs[:, self._last_action_idx:] = obs[:, self._last_action_idx:][:, self._sim_to_hardware_joint_order]
#         return obs

#     def transform_action(self, action):
#         action = action[:, self._hardware_to_sim_joint_order]
#         return action

#     def forward(self, obs):
#         transformed_obs = self.transform_observation(obs)
#         raw_action = self.actor(transformed_obs)
#         return self.transform_action(raw_action)

#     def __getitem__(self, key):
#         return self.actor[key]
    
    
class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        post_activation=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        if post_activation:
            self.actor.append(get_activation(post_activation))
        # self.actor = TransformedActor(nn.Sequential(*actor_layers))

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    

    # def load_state_dict(self, state_dict: Mapping[str, Any],
    #                     strict: bool = True, assign: bool = False):
    #     """Override load_state_dict to handle old model compatibility."""
    #     new_state_dict = {}

    #     for key in state_dict.keys():
    #         if key.startswith("actor."):  # Old format
    #             new_key = "actor.actor" + key[len("actor"):]  # Adjust for TransformedActor
    #         else:
    #             new_key = key  # Keep other keys unchanged

    #         new_state_dict[new_key] = state_dict[key]

    #     # Load the modified state dict
    #     super().load_state_dict(new_state_dict, strict=strict)

    #     print("Successfully loaded model with automatic key transformation!")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "silu":
        return nn.SiLU()
    else:
        print("invalid activation function!")
        return None
