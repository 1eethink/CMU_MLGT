from copy import Error
import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_grad_norm, mse_loss
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.distributions import FixedCategorical
from torch.distributions.kl import kl_divergence

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.use_KL_pen = True
        if self.use_KL_pen == True: 
            self.beta = 0.5

    
    def cal_value_loss(self, values, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        #TODO: Calculate the error using mse_loss (line 5)
        #Hint: If the agent is inactive, the corresponding value loss in the tensor should be 0, and the mean value function loss should take this into account.
        # from onpolicy.utils.util import get_grad_norm, mse_loss    
        
        value_loss = mse_loss(return_batch - values)
        
        
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        
        return value_loss


    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks. For a detailed description, take a look at line 80 of shared_buffer.py.
            Additionally:
                1. return_batch: Batch of returns (discounted sum of rewards)
                2. adv_targ: Batch of advantage estimates
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        _, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, old_logits_batch = sample


        old_logits_batch = check(old_logits_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)


        #TODO: Calculate Importance weights
        '''
        Hint (Calculation of Importance weights) to calculate pi_new/pi_old, you can compute the exp(log_prob_new = log_prob_old). You can refer to the evaluate_actions() function
             in rMAPPOPolicy.py on how to calculate log_prob_new. You will also notice that the function returns value function predictions and the policy distribution, both of which will 
             be required to complete the ppo_update() function.
        '''
        values, action_log_probs, dist_entropy, pi, logits = self.policy.evaluate_actions(share_obs_batch,obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, masks_batch, available_actions_batch,active_masks_batch)
        
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        
        if(self.use_KL_pen):
          #TODO: Compute L_{KLPEN}
          '''
          Hint (Calculation of KL Divergence) You may calculate KLD by first getting the old policy and current policy, and calling the  kl_divergence() method from line 8.
          To get any given policy from the output of the last layer of the actor, refer to onpolicy/algorithms/utils/distributions.py
          '''
          old_policy = FixedCategorical(logits=old_logits_batch)
          current_policy = FixedCategorical(logits=logits)
          L_KLPEN = (imp_weights * adv_targ - (self.beta * kl_divergence(old_policy, current_policy))).mean()
          
          if self._use_policy_active_masks:
              actor_loss = (-torch.sum(L_KLPEN,
                                              dim=-1,
                                              keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
          else:
              actor_loss = -torch.sum(L_KLPEN, dim=-1, keepdim=True).mean()
            
        else:
          #TODO: Compute L_{CLIP}
          term1 = imp_weights * adv_targ
          term2 = (torch.clamp(imp_weights, 1 - self.clip_param, 1 + self.clip_param)) * adv_targ
          L_CLIP = torch.min(term1, term2)
          
          if self._use_policy_active_masks:
              actor_loss = (-torch.sum(L_CLIP,
                                              dim=-1,
                                              keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
          else:
              actor_loss = -torch.sum(L_CLIP, dim=-1, keepdim=True).mean()


        policy_loss = actor_loss
        
        self.policy.actor_optimizer.zero_grad()

        # TODO: Update the actor network using stochastic gradient descent
        if update_actor == True:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        
        self.policy.actor_optimizer.step()
        
        # TODO: Update the critic network using stochastic gradient descent
        value_loss = self.cal_value_loss(values, return_batch, active_masks_batch)
        
        self.policy.critic_optimizer.zero_grad()
        
        (value_loss * self.value_loss_coef).backward()
        
        critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, policy_loss, imp_weights
    


    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):

            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            i = 0
            for sample in data_generator:
                i += 1
                value_loss, policy_loss, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

