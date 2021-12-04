from sac import PolicyNetwork
import pybullet_envs.bullet.minitaur_gym_env as e
import pybullet as p
import torch
import time

env = e.MinitaurBulletEnv(render=True)

observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# load policy network
policy = PolicyNetwork(observation_dim, action_dim, hidden_dim, device=device)
policy.load_state_dict(torch.load("./policy.pth", map_location=device))

r = []

# perform 10 walks
for i in range(10):
	state = env.reset()
	rewards = []

	# let minitaur land
	for _ in range(6):
		p.stepSimulation()

	# perform a 1000 step walk or until fallen
	for j in range(1000):
		action = policy.get_action(state).detach()
		state, reward, _, _ = env.step(action.numpy())
		rewards.append(reward)
		if env.is_fallen():
			break
		time.sleep(0.03)

	r.append(sum(rewards))

print("Average reward: ", sum(r)/10)
