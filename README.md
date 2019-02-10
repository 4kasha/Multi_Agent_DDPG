[//]: # (Image References)

[image1]: media/Tennis_early.gif "Trained Agent"

# Collaboration and Competition via MADDPG - PyTorch implementation

## Introduction

In this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

The goal of this environment is to bounce the ball back and forth between the two agents without letting the ball hit the ground.
A reward of **+0.1** is provided for the agent who hits the ball over the net. On the other hand, A reward of **-0.01** is provided for the agent whose ball hits the ground or out of bounds. 

The observation space consists of **8 variables corresponding to the position and velocity of the ball and racket**. Each agent receives its own local observation. **Two continuous actions in the range of [-1,1] are available**, corresponding to the velocity toward (or away from) the net and jumping. 

The task is episodic and in order to solve the environment your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

## Dependencies

- Python 3.6
- PyTorch 0.4.0
- ML-Agents Beta v0.4

**NOTE** : (_For Windows users_) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

## Getting Started

1. Create (and activate) a new environment with Python 3.6 via Anaconda.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name your_env_name python=3.6
	source activate your_env_name
	```
	- __Windows__: 
	```bash
	conda create --name your_env_name python=3.6 
	activate your_env_name
	```

2. Clone the repository, and navigate to the python/ folder. Then, install several dependencies (see `requirements.txt`).
    ```bash
    git clone https://github.com/4kasha/Multi_Agent_DDPG.git
    cd Multi_Agent_DDPG/python
    pip install .
    ```

3. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.

    **NOTE** : For this project, you will not need to install Unity. The link above provides you a standalone version. Also the above Tennis environment is similar to, but **not identical to** the original one on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

4. Place the file in this repository _Multi_Agent_DDPG_ and unzip (or decompress) the file.

## Instructions

- Before running code, edit parameters in `train.py`, especially you must change `env_file_name` according to your environment.
- Run the following command to get started with training your own agents!
    ```bash
    python train.py
    ```
- After finishing training weights and scores are saved in the following folder `weights` and `scores` respectively. 

## Tips

- For more details of algolithm description, hyperparameters settings and results, see [REPORT.md](REPORT.md).
- For the examples of training results, see [MARL_Results.ipynb](MARL_Results.ipynb).
- After training you can test the agent with saved weights in the folder `weights`, see [MARL_Watch_Agent.ipynb](MARL_Watch_Agent.ipynb). 
- This project is a part of Udacity's [Deep Reinforcement Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).