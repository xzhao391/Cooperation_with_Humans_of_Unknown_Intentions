import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent

from Buffer import Buffer
import torch.autograd as autograd
from scipy.sparse.linalg import cg, LinearOperator
from torch.autograd.functional import hessian
from numpy import linalg as LA
import time
def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    def __init__(self, obs_dim_list, act_dim_list, capacity, actor_lr, critic_lr, res_dir=None, device=None):
        """
        :param obs_dim_list: list of observation dimension of each agent
        :param act_dim_list: list of action dimension of each agent
        :param capacity: capacity of the replay buffer
        :param res_dir: directory where log file and all the data and figures will be saved
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f'training on device: {self.device}')
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        # create all the agents and corresponding replay buffer
        self.agents = []
        self.buffers = []
        i = 0
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            if i != 1:
                self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, self.device))
            else:
                self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, self.device))
            self.buffers.append(Buffer(capacity, obs_dim, act_dim, self.device))
            i += 1
        if res_dir is None:
            self.logger = setup_logger('maddpg.log')
        else:
            self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, actions, rewards, next_obs, dones):
        """add experience to buffer"""
        for n, buffer in enumerate(self.buffers):
            buffer.add(obs[n], actions[n], rewards[n], next_obs[n], dones[n])

    def sample(self, batch_size, agent_index):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs_list, act_list, next_obs_list, next_act_list = [], [], [], []
        reward_cur, done_cur, obs_cur = None, None, None
        for n, buffer in enumerate(self.buffers):
            obs, action, reward, next_obs, done = buffer.sample(indices)
            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[n].target_action(next_obs).clone().detach().requires_grad_(True))
            if n == agent_index:  # reward and done of the current agent
                obs_cur = obs
                reward_cur = reward
                done_cur = done

        return obs_list, act_list, reward_cur, next_obs_list, done_cur, next_act_list, obs_cur

    def select_action(self, obs):
        actions = []
        for n, agent in enumerate(self.agents):  # each agent select action according to their obs
            o = torch.from_numpy(obs[n]).unsqueeze(0).float().to(self.device)  # torch.Size([1, state_size])
            # Note that the result is tensor, convert it to ndarray before input to the environment
            act = agent.action(o).squeeze(0).detach().cpu().numpy()
            actions.append(act)
            # self.logger.info(f'agent {n}, obs: {obs[n]} action: {act}')
        return actions

    def learn(self, batch_size, gamma):
        # start = time.time()
        adv_eps_s = 1e-4
        adv_eps = 1e-3
        for i, agent in enumerate(self.agents):
            if i == 0:
                continue
            elif i == 1:
                friend = 2
                adv = 0
            else:
                friend = 1
                adv = 0

            obs, act, reward_cur, next_obs, done_cur, next_act, obs_cur = self.sample(batch_size, i)
            #Gradient for Critic
            next_target_critic_initial = agent.target_critic_value(next_obs, next_act)
            p1 = next_act[friend]
            p2 = next_act[adv]
            f1 = next_target_critic_initial.mean()

            grad_p1_critic = self.compute_grad2(f1, p1, p2)
            next_action= next_act
            next_action[friend] = p1 + adv_eps_s*grad_p1_critic*LA.norm(p1.cpu().detach().numpy())
            f1_new = agent.target_critic_value(next_obs, next_action).mean()
            D2f2_critic = autograd.grad(-f1_new, p2, retain_graph=True, create_graph=True)[0]
            next_action[adv] = p2 + adv_eps*D2f2_critic * LA.norm(p2.cpu().detach().numpy())
            next_target_critic_value = agent.target_critic_value(next_obs, next_action)

            critic_value = agent.critic_value(obs, act)
            target_value = reward_cur + gamma * next_target_critic_value * (1 - done_cur)
            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            #Gradient for actor
            action, logits =  agent.action(obs_cur, model_out=True)
            act[i] = action
            actor_loss_initial = agent.critic_value(obs, act)
            p1 = act[friend]
            p2 = act[adv]
            f1 = actor_loss_initial.mean()
            grad_p1_actor = self.compute_grad2(f1, p1, p2)
            action_actor= act
            action_actor[friend] = p1 + adv_eps_s*grad_p1_actor*LA.norm(p1.cpu().detach().numpy())
            f1_actor = agent.critic_value(next_obs, action_actor).mean()
            D2f2_actor = autograd.grad(-f1_actor, p2, retain_graph=True, create_graph=True)[0]
            action_actor[adv] = p2 + adv_eps*D2f2_actor * LA.norm(p2.cpu().detach().numpy())
            actor_loss = -agent.critic_value(obs, action_actor).mean()

            # update actor
            # action of the current agent is calculated using its actor
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def compute_grad2(self, f1, p1, p2):
        D1f1 = autograd.grad(f1, p1, retain_graph=True, create_graph=True)[0]
        # D1f1_vec = torch.cat([g.contiguous().view(-1) for g in D1f1])

        D2f1 = autograd.grad(f1, p2, retain_graph=True, create_graph=True)[0]
        D2f1_vec = torch.cat([g.contiguous().view(-1) for g in D2f1])
        D2f2 = autograd.grad(-f1, p2, retain_graph=True, create_graph=True)[0]
        D2f2_vec = torch.cat([g.contiguous().view(-1) for g in D2f2])

        def D12f2(vec):
            """
            input:  numpy array
            output: torch tensor
            """
            vec = torch.Tensor(vec).cuda()
            _Avec = autograd.grad(D2f2_vec, p1, vec, retain_graph=True)
            return torch.cat([g.contiguous().view(-1) for g in _Avec])

        def D22f2_matvec(vec):
            """
            input:  numpy array
            output: numpy array
            """
            reg = 1e-6
            vec = torch.Tensor(vec).cuda()
            _Avec = autograd.grad(D2f2_vec, p2, vec, retain_graph=True)
            Avec = torch.cat([g.contiguous().view(-1) for g in _Avec])
            Avec += reg * vec
            return np.array(Avec.cpu())

        D22f2_lo = LinearOperator(shape=(5120, 5120), matvec=D22f2_matvec)
        w, _ = cg(D22f2_lo, D2f1_vec.cpu().detach().numpy(), maxiter=10)  # D22f2^-1 * D2f1
        grad_imp = D12f2(w)
        grad_p1 = D1f1 - torch.reshape(grad_imp, (1024, 5))
        grad_p1 = grad_p1/LA.norm(grad_p1.cpu().detach().numpy())
        return grad_p1

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)


        # next_act_origin = next_act
        # D2f2 = autograd.grad(-f1, p2, retain_graph=True, create_graph=True)[0]
        # D1f1 = autograd.grad(f1, p1, retain_graph=True, create_graph=True)[0]
        # next_act_origin[GoodForNow] = p1 + D1f1 / LA.norm(D1f1.cpu().detach().numpy()) * LA.norm(p1.cpu().detach().numpy())
        # next_act_origin[0] = p2 + D2f2 / LA.norm(D2f2.cpu().detach().numpy()) * LA.norm(p2.cpu().detach().numpy())
        # next_target_critic_value_origin = agent.target_critic_value(next_obs, next_act_origin)
        # print(next_target_critic_value.mean() - next_target_critic_value_origin.mean())

        # act_origin = act
        # D2f2 = autograd.grad(-f1, p2, retain_graph=True, create_graph=True)[0]
        # D1f1 = autograd.grad(f1, p1, retain_graph=True, create_graph=True)[0]
        # act_origin[GoodForNow] = p1 + D1f1 / LA.norm(D1f1.cpu().detach().numpy()) * LA.norm(p1.cpu().detach().numpy())
        # act_origin[0] = p2 + D2f2 / LA.norm(D2f2.cpu().detach().numpy()) * LA.norm(p2.cpu().detach().numpy())
        # actor_loss_origin = -agent.critic_value(obs, act_origin).mean()
        # print(actor_loss_origin-actor_loss)

        # if i < 1:
        #     adv_rate = [adv_eps_s * (j < 1) + adv_eps * (j >= 1) for j in range(3)]
        # else:
        #     adv_rate = [adv_eps_s * (j > 1) + adv_eps * (j <= 1) for j in range(3)]
        # adv_rate[i] = 0
