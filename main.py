import time
from collections import deque
from typing import Any, Dict

# RL imports
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Autoparameter tuning imports
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

#image processing imports
import cv2

# other ml library
import numpy as np
import torch
import torch.nn as nn

# file import
from gym_environment_wrapper import ConcateObs
from auto_hyparameter_tuner import TrialEvalCallback, sample_dqn_params, Objective_function


ENV_NAME = 'SpaceInvaders-v4'
NUM_FRAMES = 4 # number of concatenated frames
NUM_EVAL_EPISODES = 100 # number of evaluation episodes
NUM_EVAL_ENV = 5 # number of evaluation environment
BUDGET = 100000 # time step budget
SEED = 0

N_TRIALS = 100 # maximum number of trials
N_JOBS = 1 # number of jobs to run in parallels
N_STARTUP_TRIALS = 5 # do random sampling for these number of trials
N_EVALUATIONS = 2 # not sure about this parameter
EVAL_FREQ = int(BUDGET/N_EVALUATIONS)
TIMEOUT = int(60 * 30) # 30 minutes

DEFAULT_HYPERPARAMS = {
    "policy": "CnnPolicy",
    "buffer_size": 60000,
    "seed": 0
}

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    wrapped_env = ConcateObs(env, NUM_FRAMES, normalize_frame=False)

    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    objective_function = Objective_function(
            default_hyperparams=DEFAULT_HYPERPARAMS,
            sampler_function=sample_dqn_params,
            rl_enviroment=wrapped_env,
            env_name=ENV_NAME,
            num_eval_env=NUM_EVAL_ENV,
            num_eval_episodes=NUM_EVAL_EPISODES,
            eval_freq=EVAL_FREQ,
            time_budget=BUDGET,
            model=DQN,
            trial_call_back=TrialEvalCallback,
            wrapper_class=ConcateObs,
            wrapper_kwargs=dict(
                k=NUM_FRAMES,
                normalize_frame=False
            ),
            seed=0,
            eval_deterministic=True
    )

    try:
        study.optimize(objective_function, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()
