#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorageAMP

class PPOAMP(PPO):    
    def __init__(self, *args, discriminator, obs_demo, discriminator_l2_reg, discriminator_grad_penalty, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.obs_demo = obs_demo
        self.discriminator_l2_reg = discriminator_l2_reg
        self.discriminator_grad_penalty = discriminator_grad_penalty
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=kwargs['learning_rate'])

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, amp_obs_shape, action_shape):
        self.storage = RolloutStorageAMP(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, amp_obs_shape, action_shape, self.device
        )

    def act(self, obs, critic_obs, amp_obs):
        self.transition.amp_observations = amp_obs
        return super().act(obs, critic_obs)

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss
   
    def _train_discriminator(self):
        self.discriminator.train()
        
        value_losses = []
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(num_mini_batches=1, num_epochs=self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(num_mini_batches=1, num_epochs=self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            amp_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            idx = torch.randint(0, self.obs_demo.size(0), (obs_batch.size(0),))
            
            obs_demo = torch.autograd.Variable(self.obs_demo[idx], requires_grad=True).to(self.device)
            
            disc_demo_logit = self.discriminator(obs_demo)
            disc_agent_logit = self.discriminator(amp_obs_batch)
            
            # Classic Discriminator Loss
            disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
            disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
            disc_loss_p1 = 0.5 * (disc_loss_agent + disc_loss_demo)
            
            # Discriminator weight regularization
            disc_logit_layer_weight = self.discriminator[0].weight # weight of the first layer. 
            disc_logit_reg_loss = torch.sum(torch.square(disc_logit_layer_weight)) # L2 regularization
            
            # gradient penalty 
            disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                                create_graph = True, retain_graph = True, only_inputs = True)[0]
            disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
            disc_grad_penalty = torch.mean(disc_demo_grad)
                    
            # disc_loss = disc_loss_p1 + 0.01 * disc_logit_reg_loss + 5 * disc_grad_penalty
            # disc_loss = disc_loss_p1 + 0.0 * disc_logit_reg_loss + 0.0 * disc_grad_penalty
            disc_loss = disc_loss_p1 + self.discriminator_l2_reg * disc_logit_reg_loss + self.discriminator_grad_penalty * disc_grad_penalty
            self.optimizer_discriminator.zero_grad()
            disc_loss.backward()
            self.optimizer_discriminator.step()
            
            value_losses.append(disc_loss.item())
            
        return np.mean(value_losses)
    

    def update(self, train_discriminator=False):
        if train_discriminator:
            discriminator_loss = self._train_discriminator()
        else:
            discriminator_loss = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            amp_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, discriminator_loss
