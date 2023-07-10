import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.autograd as autograd
import copy

from treeqn.utils.treeqn_utils import discount_with_dones, build_sequences, get_paths, get_subtree, time_shift_tree, \
    ReplayBufferElement
from treeqn.utils.schedule import LinearSchedule
from treeqn.utils.pytorch_utils import cudify

USE_CUDA = torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def make_variable(data, *args, **kwargs):
    v = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        v = v.cuda()
    return v


class Learner(object):
    def __init__(self, model,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, number_updates=int(1e6), lrschedule='linear',
                 use_actor_critic=False, rew_loss_coef=0.0, st_loss_coef=0.0,
                 subtree_loss_coef=0.0,
                 nsteps=5, nenvs=1,
                 tree_depth=0):
        self.max_grad_norm = max_grad_norm
        self.use_actor_critic = use_actor_critic
        self.use_reward_loss = model.predict_rewards and rew_loss_coef > 0
        self.rew_loss_coef = rew_loss_coef
        self.use_st_loss = st_loss_coef > 0 and tree_depth > 0
        self.st_loss_coef = st_loss_coef
        self.subtree_loss_coef = subtree_loss_coef
        self.use_subtree_loss = subtree_loss_coef > 0
        self.model = model
        self.nsteps = nsteps
        self.nenvs = nenvs
        self.batch_size = nsteps * nenvs
        self.num_actions = model.num_actions
        self.tree_depth = tree_depth

        if USE_CUDA:
            self.model = self.model.cuda()

        if not self.use_actor_critic:
            self.target_model = copy.deepcopy(self.model)

            if USE_CUDA:
                self.target_model = self.target_model.cuda()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=alpha, eps=epsilon)

        if lrschedule == "linear":
            self.scheduler = LambdaLR(self.optimizer, lambda step: 1.0 - (step / number_updates))
        elif lrschedule == "constant":
            self.scheduler = LambdaLR(self.optimizer, lambda step: 1.0)
        else:
            raise ValueError("lrschedule should be 'linear' or 'constant'")

        self.step = self.model.step
        if self.use_actor_critic:
            self.value = self.model.value
            self.ent_coef = ent_coef
            self.vf_coef = vf_coef
        else:
            self.value = self.target_model.value

    def train(self, obs, next_obs, returns, rewards, masks, actions, values):
        """
        :param obs: [batch_size x height x width x channels] observations in NHWC
        :param next_obs: [batch_size x height x width x channels] one-step next states
        :param returns: [batch_size] n-step discounted returns with bootstrapped value
        :param rewards: [batch_size] 1-step rewards
        :param masks: [batch_size] boolean episode termination mask
        :param actions: [batch_size] actions taken
        :param values: [batch_size] predicted state values
        """

        # compute the sequences we need to get back reward predictions
        action_sequences, reward_sequences, sequence_mask = build_sequences(
            [torch.from_numpy(actions), torch.from_numpy(rewards)], masks, self.nenvs, self.nsteps, self.tree_depth, return_mask=True)
        action_sequences = cudify(action_sequences.long().squeeze(-1))
        reward_sequences = make_variable(reward_sequences.squeeze(-1))
        sequence_mask = make_variable(sequence_mask.squeeze(-1))

        Q, V, tree_result = self.model(obs)

        actions = make_variable(torch.from_numpy(actions).long(), requires_grad=False)
        returns = make_variable(torch.from_numpy(returns), requires_grad=False)

        policy_loss, value_loss, reward_loss, state_loss, subtree_loss_np, policy_entropy = 0, 0, 0, 0, 0, 0
        if self.use_actor_critic:
            values = make_variable(torch.from_numpy(values), requires_grad=False)
            advantages = returns - values
            probs = F.softmax(Q, dim=-1)
            log_probs = F.log_softmax(Q, dim=-1)
            log_probs_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            pg_loss = -torch.mean(log_probs_taken * advantages.squeeze())
            vf_loss = F.mse_loss(V, returns)
            entropy = -torch.mean(torch.sum(probs * log_probs, 1))
            loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

            policy_loss = pg_loss.data.cpu().numpy()
            value_loss = vf_loss.data.cpu().numpy()
            policy_entropy = entropy.data.cpu().numpy()
        else:
            Q_taken = Q.gather(1, actions.unsqueeze(1)).squeeze()
            bellman_loss = F.mse_loss(Q_taken, returns)
            loss = bellman_loss
            value_loss = bellman_loss.data.cpu().numpy()

        if self.use_reward_loss:
            r_taken = get_paths(tree_result["rewards"], action_sequences, self.batch_size, self.num_actions)
            rew_loss = F.mse_loss(torch.cat(r_taken, 1), reward_sequences, reduce=False)
            rew_loss = torch.sum(rew_loss * sequence_mask) / sequence_mask.sum()
            loss = loss + rew_loss * self.rew_loss_coef
            reward_loss = rew_loss.data.cpu().numpy()

        if self.use_st_loss:
            st_embeddings = tree_result["embeddings"][0]
            st_targets, st_mask = build_sequences([st_embeddings.data], masks, self.nenvs, self.nsteps, self.tree_depth, return_mask=True, offset=1)
            st_targets = make_variable(st_targets.view(self.batch_size, -1))
            st_mask = make_variable(st_mask.view(self.batch_size, -1))

            st_taken = get_paths(tree_result["embeddings"][1:], action_sequences, self.batch_size, self.num_actions)

            st_taken_cat = torch.cat(st_taken, 1)

            st_loss = F.mse_loss(st_taken_cat, st_targets, reduce=False)
            st_loss = torch.sum(st_loss * st_mask) / st_mask.sum()

            state_loss = st_loss.data.cpu().numpy()
            loss = loss + st_loss * self.st_loss_coef

        if self.use_subtree_loss:
            subtree_taken = get_subtree(tree_result["values"], action_sequences, self.batch_size, self.num_actions)
            target_subtrees = tree_result["values"][0:-1]
            subtree_taken_clip = time_shift_tree(subtree_taken, self.nenvs, self.nsteps, -1)
            target_subtrees_clip = time_shift_tree(target_subtrees, self.nenvs, self.nsteps, 1)

            subtree_loss = [(s_taken - s_target).pow(2).mean() for (s_taken, s_target) in zip(subtree_taken_clip, target_subtrees_clip)]
            subtree_loss = sum(subtree_loss)
            subtree_loss_np = subtree_loss.data.cpu().numpy()

            loss = loss + subtree_loss * self.subtree_loss_coef

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return policy_loss, value_loss, reward_loss, state_loss, subtree_loss_np, policy_entropy, grad_norm


def compute_returns(rewards, dones, last_value, gamma):
    rewards = rewards.tolist()
    dones = dones.tolist()
    if dones[-1] == 0:
        returns = discount_with_dones(rewards + [last_value], dones + [0], gamma)[:-1]
    else:
        returns = discount_with_dones(rewards, dones, gamma)

    return returns


class Runner(object):
    def __init__(self, env, learner, nsteps=5, nstack=4, gamma=0.99, obs_dtype=np.uint8, eps_million_frames=4, buffer_size=100000):
        self.env = env
        self.learner = learner
        nh, nw, nc = env.observation_space.shape
        self.nc = nc
        self.nenv = env.num_envs
        self.batch_ob_shape = (self.nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((self.nenv, nh, nw, nc * nstack), dtype=obs_dtype)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.obs_dtype = obs_dtype
        self.dones = [False for _ in range(self.nenv)]

        self.frames_counter = 0
        if not self.learner.use_actor_critic:
            self.eps_schedule = LinearSchedule(eps_million_frames*1e6, 0.05)

        self.buffer_max_size = math.ceil(buffer_size / self.nsteps)
        self.buffer_size = 0
        self.index = 0
        self.replay_buffer = [None] * self.buffer_max_size

    def get_buffer_size(self):
        return self.buffer_size * self.nsteps

    def _add(self, buffer_elements):
        for element in buffer_elements:
            self.replay_buffer[self.index] = element
            self.index = (self.index + 1) % self.buffer_max_size
            self.buffer_size = min(self.buffer_max_size, self.buffer_size + 1)

    def sample(self, batch_size):
        n_elements = math.ceil(batch_size / self.nsteps)
        buffer_elements = random.sample(self.replay_buffer[:self.buffer_size], n_elements)

        last_obs = np.stack([element.last_observation for element in buffer_elements])
        last_values = self.learner.value(last_obs).detach().cpu().numpy()
        mb_returns = []
        mb_obs = []
        mb_rewards = []
        mb_masks = []
        mb_actions = []
        for element, last_value in zip(buffer_elements, last_values):
            mb_returns.append(np.asarray(compute_returns(element.rewards, element.dones[1:], last_value, self.gamma)))
            mb_obs.append(element.observations)
            mb_rewards.append(element.rewards)
            mb_masks.append(element.dones[:-1])
            mb_actions.append(element.actions)

        mb_obs = np.concatenate(mb_obs).reshape(self.batch_ob_shape)
        mb_returns = np.concatenate(mb_returns).flatten().astype(np.float32)
        mb_rewards = np.concatenate(mb_rewards).flatten()
        mb_actions = np.concatenate(mb_actions).flatten()
        mb_masks = np.concatenate(mb_masks).flatten()
        return mb_obs, mb_returns, mb_rewards, mb_masks, mb_actions

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs[:, :, :, :]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_dones = [], [], [], []
        if not self.learner.use_actor_critic:
            self.learner.model.eps_threshold = self.eps_schedule.value(self.frames_counter)
        for n in range(self.nsteps):
            actions, values = self.learner.step(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.dones = dones
            for m, done in enumerate(dones):
                if done:
                    self.obs[m] = self.obs[m] * 0
            self.update_obs(obs)
            mb_rewards.append(rewards)
            self.frames_counter += self.nenv
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)

        buffer_elements = []
        for i in range(self.nenv):
            replay_buffer_element = ReplayBufferElement(mb_obs[i], mb_actions[i], mb_rewards[i], mb_dones[i], self.obs[i].copy())
            buffer_elements.append(replay_buffer_element)

        self._add(buffer_elements)
