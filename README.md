# RL-Walking
Third Year university project. Using reinforcement learning to teach a quadrupedal robot to walk.

sac.py contains the classes and functions needed to run a SAC1 and SAC2 learning environment<br>
train.py trains a minitaur walking with the SAC2 algorithm for 500,000 total environment steps<br>
test.py loads a pre-trained policy and performs 10 1000 step walks using the pre-trained policy<br>
policy.pth is a sample policy learned by SAC2 which is run in test.py

See https://pypi.org/project/pybullet/ and https://pybullet.org/ for correct pybullet installation
