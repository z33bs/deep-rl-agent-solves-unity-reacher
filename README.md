# Deep learning agent controls 2 armed robot arm

This repo solves the 'Reacher' environment as part of Udacity's Deep Reinforcement Nanodegree.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm (continuous state space). Each action is a vector with four numbers (continuous action space), corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

![reacher environment](reacher.gif)



## Installation

The following should get you going:

#### Download and install Reacher environment

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

#### Install dependencies

```bash
# create conda virtual environment
conda create --yes -n reacher python=3.6
conda activate reacher
# install dependencies
pip install -r requirements.txt 
```

#### Train

To test everything is working first try `python test.py`. This will run for one episode with random actions.

If all looks good run with default parameters (on OSX):

```bash
python agent_runner.py
```

for other operating systems, run with the name of your environment file that you downloaded:

```bash
python agent_runner.py --env '<environment filename>'
```

For runner options use the `--help` flag. For example, you can view the trained agent with

#### Test

```bash
python agent_runner.py --mode 'test' --load 'final'
```

#### Plot

Plot training scores from logfile with

```bash
python agent_runner.py --mode 'plot' --load 'final_agent_log.txt'
```



## Results

See **[REPORT](REPORT.md)** for details. Figure below plots the score over time.

![score over time](score.png)

