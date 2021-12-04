import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import random
import numpy as np
import pybullet as p


class ReplayMemory(object):
	"""
	Finite list object
	"""
	def __init__(self, capacity):
		self.capacity = int(capacity)
		self.memory = []
		self.position = 0

	def push(self, state, action, next_state, reward, done):
		"""
		Save a state transition to the memory
		:param state: State in the environment
		:param action: Action applied to the environment
		:param next_state: State after applying action
		:param reward: Reward achieved by applying action
		:param done: Boolean denoting whether an end state has been reached
		:return: None
		"""
		# Add a new position in memory if the current size is less than capacity
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = (state, action, next_state, reward, done)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		"""
		Returns a random sample of the memory of size batch_size
		:param batch_size: How many items of the memory to return
		:return: a sample of the replay memory
		"""
		if batch_size >= self.capacity:
			batch = self.memory
		else:
			batch = random.sample(self.memory, batch_size)
		sample = map(np.stack, zip(*batch))
		return sample

	def __len__(self):
		return len(self.memory)


class SoftQNetwork(nn.Module):
	"""
	Soft Q Neural Network
	"""
	def __init__(self, state_dim, action_dim, hidden_size=256, init_w=3e-3):
		super(SoftQNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state, action):
		"""
		Feeds a state and action forward through the soft Q network
		:param state: state to be fed through the network, float tensor
		:param action: action to be fed through the network, float tensor
		:return: Q value of the state and action, float
		"""
		x = torch.cat([state, action], 1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class ValueNetwork(nn.Module):
	"""
	Value Network
	Only used in SACAgent, replaced by entropy temperature in SAC2Agent
	"""
	def __init__(self, state_dim, hidden_dim, init_w=3e-3):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)

	def forward(self, state):
		"""
		Feeds a state forward through the value network
		:param state: state to be fed through the network, float tensor
		:return: value of the state , float
		"""
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class PolicyNetwork(nn.Module):
	"""
	Gaussian Policy network
	"""
	def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, device="cpu"):
		super(PolicyNetwork, self).__init__()

		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.linear1 = nn.Linear(state_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)

		self.mean_linear = nn.Linear(hidden_size, action_dim)
		self.mean_linear.weight.data.uniform_(-init_w, init_w)
		self.mean_linear.bias.data.uniform_(-init_w, init_w)

		self.log_std_linear = nn.Linear(hidden_size, action_dim)
		self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		self.log_std_linear.bias.data.uniform_(-init_w, init_w)

		self.device = device

	def forward(self, state):
		"""
		Feeds a state forward through the policy network
		:param state: state to be fed through the network, float tensor
		:return: mean and standard deviation of the probability distribution of action given state, tensor
		"""
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))

		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

		return mean, log_std

	def evaluate(self, state, epsilon=1e-6):
		"""
		Get an action and the log of the probability of that action given state
		Used for calculating loss functions
		:param state: Environment state, float tensor
		:param epsilon: noise
		:return: action: environment action, float tensor
				log_prob: log(pi(action | state)), float
		"""
		mean, log_std = self.forward(state)
		std = log_std.exp()

		# Sample an action from the gaussian distribution with the mean and std
		normal = Normal(0, 1)
		z = normal.sample()
		action = torch.tanh(mean + std * z.to(self.device))

		# Get the log of the probability of action plus some noise
		log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)

		return action, log_prob

	def get_action(self, state):
		"""
		Get an action given state. Used in training
		:param state: environment state float tensor
		:return: action: float tensor
		"""
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		mean, log_std = self.forward(state)
		std = log_std.exp()

		# Sample an action from the gaussian distribution with the mean and std
		normal = Normal(0, 1)
		z = normal.sample().to(self.device)
		action = torch.tanh(mean + std * z)

		action = action.cpu()
		return action[0]


class SACAgent:
	"""
	An agent for the first generation of Soft Actor Critic learning algorithm
	Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
	Haarnoja et al. 2018
	"""
	def __init__(self, env, lr=3e-4, replay_buffer_size=1000000):
		"""
		:param env: an instance of an OpenAI Gym environment that is being learned on.
		:param lr: float, the learning rate used to update the parameters
		:param replay_buffer_size: int, size of the replay buffer
		"""

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		try:
			action_dim = env.action_space.shape[0]
		except IndexError:
			action_dim = env.action_space.n
		try:
			observation_dim = env.observation_space.shape[0]
		except IndexError:
			observation_dim = env.observation_space.n

		self.value_net = ValueNetwork(observation_dim, 256).to(self.device)
		self.target_value_net = ValueNetwork(observation_dim, 256).to(self.device)

		self.soft_q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
		self.soft_q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)

		self.policy = PolicyNetwork(observation_dim, action_dim, 256, device=self.device).to(self.device)

		self.target_value_net.load_state_dict(self.value_net.state_dict())

		self.value_criterion = nn.MSELoss()
		self.q1_criterion = nn.MSELoss()
		self.q2_criterion = nn.MSELoss()

		self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)
		self.q1_optim = optim.Adam(self.soft_q1.parameters(), lr=lr)
		self.q2_optim = optim.Adam(self.soft_q1.parameters(), lr=lr)
		self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

		self.mem_size = replay_buffer_size
		self.replay_buffer = ReplayMemory(self.mem_size)

	def update(self, batch_size, gamma=0.99, tau=1e-2):
		"""
		Update the parameters of the agent
		:param batch_size: Size of sample taken from replay memory
		:param gamma: Discount factor for calculating Q loss
		:param tau: Smoothing rate for updating target functions
		:return: None
		"""

		state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)

		state = torch.FloatTensor(state).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		action = torch.FloatTensor(action).to(self.device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
		done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

		pred_q1 = self.soft_q1(state, action)
		pred_q2 = self.soft_q2(state, action)
		pred_val = self.value_net(state).mean()
		new_action, log_prob = self.policy.evaluate(next_state)

		# Train Q using Q loss:
		# J(Q_i) = 1/2 (Q_i(s_t, a_t) - Q'(s_t, a_t))^2, where,
		# Q'(s_t, a_t) = r(s_t, a_t) + gamma * target_V(s_(t+1))
		target_val = self.target_value_net(next_state)
		target_q = reward + (1 - done) * gamma * target_val
		q1_loss = self.q1_criterion(pred_q1, target_q.detach())
		q2_loss = self.q2_criterion(pred_q2, target_q.detach())
		self.q1_optim.zero_grad()
		q1_loss.backward()
		self.q1_optim.step()
		self.q2_optim.zero_grad()
		q2_loss.backward()
		self.q2_optim.step()

		# Train V with the loss function
		# J(V) = 1/2 (V(s_t) - (Q(s_t, a_t) - log policy(a_t, s_t)))^2
		pred_new_q = torch.min(self.soft_q1(state, new_action), self.soft_q2(state, new_action))
		target_val_func = pred_new_q - log_prob
		val_loss = self.value_criterion(pred_val, target_val_func.detach())
		self.value_optim.zero_grad()
		val_loss.backward()
		self.value_optim.step()

		# Train Policy with loss function
		# J(policy) = mean(log policy(a_t, s_t) - Q(s_t, a_t))
		policy_loss = (log_prob - pred_new_q).mean()
		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		# Update the target value parameters with
		# target_param = tau * param + (1 - tau) * target_param
		for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
			target_param.data.copy_(
				param.data * tau + target_param.data * (1.0 - tau)
			)

	def save_policy(self, path):
		"""
		Saves the state dictionary of the policy network the the specified path
		:param path: The path to save the state dictionary to
		:return: None
		"""
		torch.save(self.policy.state_dict(), path)


class SAC2Agent:
	"""
	Agent for the second generation of the Soft Actor Critic learning algorithm presented in
	Soft Actor-Critic Algorithms and Applications, Haarnoja et al. 2018
	Differs from SACAgent by replacing the need for a value function with an entropy temperature parameter and tuning
	this while learning
	"""
	def __init__(self, env, alpha=0.2, alr=3e-4, qlr=3e-4, policy_lr=3e-4, mem_size=1e6):
		"""
		:param env: an instance of an OpenAI Gym environment that is being learned on.
		:param alpha: float, initial value for alpha
		:param alr: float, the learning rate used to update alpha
		:param qlr: float, the learning rate used to update the q functions
		:param policy_lr: float, the learning rate used to update the policy function
		:param mem_size: int, size of the replay buffer, 1e6 by default
		"""

		try:
			action_dim = env.action_space.shape[0]
		except IndexError:
			action_dim = env.action_space.n
		try:
			observation_dim = env.observation_space.shape[0]
		except IndexError:
			observation_dim = env.observation_space.n

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
		self.q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)

		self.target_q1 = SoftQNetwork(observation_dim, action_dim).to(self.device)
		self.target_q2 = SoftQNetwork(observation_dim, action_dim).to(self.device)
		self.target_q1.load_state_dict(self.q1.state_dict())
		self.target_q2.load_state_dict(self.q2.state_dict())

		self.policy = PolicyNetwork(observation_dim, action_dim, 256, device=self.device).to(self.device)

		self.alpha = alpha
		self.target_a = -action_dim
		self.log_a = torch.zeros(1, requires_grad=True, device=self.device)

		self.q1_criterion = nn.MSELoss()
		self.q2_criterion = nn.MSELoss()

		self.q1_optim = optim.Adam(self.q1.parameters(), lr=qlr)
		self.q2_optim = optim.Adam(self.q2.parameters(), lr=qlr)
		self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
		self.a_optim = optim.Adam([self.log_a], lr=alr)

		self.mem_size = mem_size
		self.replay_buffer = ReplayMemory(mem_size)

	def update(self, batch_size, gamma=0.99, tau=5e-3):
		"""
		Update the parameters of the agent
		:param batch_size: Size of sample taken from replay memory
		:param gamma: Discount factor for calculating Q loss
		:param tau: Smoothing rate for updating target functions
		:return: None
		"""

		state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)

		# Convert all to tensors
		state = torch.FloatTensor(state).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		action = torch.FloatTensor(action).to(self.device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
		done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
		next_action, next_log_prob = self.policy.evaluate(next_state)

		# Update Q networks using the loss function
		# J(Q_i) = 1/2 (Q(s_t, a_t) - (r(s_t, a_t) + gamma * V(s_(t+1)))^2  where,
		# V(s_t) = Q(s_t, a_t) - alpha * log policy(a_t, s_t)
		next_q1 = self.target_q1(next_state, next_action)
		next_q2 = self.target_q2(next_state, next_action)
		value = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
		expected_q = reward + gamma * (1 - done) * value

		q1 = self.q1(state, action)
		q2 = self.q2(state, action)
		q1_loss = self.q1_criterion(q1, expected_q.detach())
		q2_loss = self.q2_criterion(q2, expected_q.detach())
		self.q1_optim.zero_grad()
		q1_loss.backward()
		self.q1_optim.step()
		self.q2_optim.zero_grad()
		q2_loss.backward()
		self.q2_optim.step()

		# Update policy network with loss function
		# J(policy) = alpha * log policy(a_t, s_t) - Q(s_t, a_t)
		new_action, log_prob = self.policy.evaluate(state)
		policy_loss = (self.alpha * log_prob - torch.min(self.q1(state, new_action), self.q2(state, new_action))).mean()
		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		# Update temperature with loss function
		# J(alpha) = -alpha * log policy(a_t, s_t) - alpha * target_alpha
		alpha_loss = (self.log_a * (-log_prob - self.target_a).detach()).mean()
		self.a_optim.zero_grad()
		alpha_loss.backward()
		self.a_optim.step()
		self.alpha = self.log_a.exp()

		# Update target networks
		# target_param = tau * param + (1 - tau) * target_param
		for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
			target_param.data.copy_(
				tau * param.data + (1 - tau) * target_param.data
			)
		for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
			target_param.data.copy_(
				tau * param.data + (1 - tau) * target_param.data
			)

	def save_policy(self, path):
		"""
		Saves the state dictionary of the policy network the the specified path
		:param path: The path to save the state dictionary to
		:return: None
		"""
		torch.save(self.policy.state_dict(), path)


def train_loop(env, agent, max_total_steps, max_steps, batch_size, intermediate_policies=False, path="./", verbose=False, update_all=True):
	"""
	Training loop
	:param env: Instance of OpenAI gym environment
	:param agent: Instance of SACAgent or SAC2Agent
	:param max_total_steps: int, Maximum number of environment steps taken during training
	:param max_steps: int, Maximum number of steps in each episode
	:param batch_size: int, Size of sample taken from replay memory
	:param intermediate_policies: Bool if you want 20, 40, 60, 80% policy saved. False by default
	:param path: String, Where to save intermediate policies. './' by default
	:param verbose: Bool prints progress at 1% increments. False by default
	:param update_all: Bool whether to update after each environment step. True by default
	:return: list of rewards achieved in training
	"""

	rewards = []
	steps = 0

	while steps < max_total_steps:
		state = env.reset()
		ep_reward = 0

		# Step the simulation 5 to stop learning starting midair if it is the minitaur env
		try:
			env.minitaur
			for i in range(5):
				p.stepSimulation()
		except AttributeError:
			continue

		for step in range(max_steps):
			if verbose and not (steps % (max_total_steps // 100)):
				print("Steps: {}".format(steps))

			# Get random action until the replay memory has been filled, then get action from policy network
			if steps > 2 * batch_size:
				action = agent.policy.get_action(state).detach()
				next_state, reward, done, _ = env.step(action.numpy())
			else:
				action = env.action_space.sample()
				next_state, reward, done, _ = env.step(action)

			# Add state action transition to replay memory
			agent.replay_buffer.push(state, action, next_state, reward, done)

			state = next_state
			ep_reward += reward
			steps += 1

			if update_all:
				if len(agent.replay_buffer) > batch_size:
					agent.update(batch_size)

			else:
				if len(agent.replay_buffer) > batch_size and not steps % 10:
					agent.update(batch_size)

			# Save the policy network at 20% increments
			if intermediate_policies and not steps % (max_total_steps // 5):
				agent.save_policy(path + "policy{}.pth".format((steps // (max_total_steps // 5))))

			# Break out of loop if an end state has been reached
			if done:
				break

		rewards.append(ep_reward)

	return rewards
