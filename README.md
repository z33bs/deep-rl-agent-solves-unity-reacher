# ppo-agent-solves-robot-arm

## Installation

```
pip install --upgrade pip
pip install unityagents==0.4.0

conda create -n reacher python=3.6 pytorch

conda activate reacher
# pip install mlagents==0.25.0
pip install mlagents_envs==0.25.0
git clone https://github.com/Unity-Technologies/ml-agents.git
pip install --editable ./ml-agents/
```

Download environment file for your OS

```
conda env export --from-history
conda list -e
conda env remove -n reacher
```

```
conda create --yes -n reacher python=3.6
conda activate reacher
pip install -r requirements.txt 

# 
python test.py 
```

```

```

