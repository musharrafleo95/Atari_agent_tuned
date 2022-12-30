from typing import Any, Dict
import optuna
import torch.nn as nn

# defining hyperparameter sampler function
def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # buffer_size = trial.suggest_int("buffer_size", 100000, 130000)
    batch_size = trial.suggest_int("batch_size", 8, 256, log=True)

    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
#     net_arch = trial.suggest_categorical("net_arch",  ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn",  ["tanh", "relu"])


    # Display true values
    trial.set_user_attr("gamma_", gamma)
#     trial.set_user_attr("n_steps", n_steps)

#     # Custom actor (pi) and value function (vf) networks
#     # of two layers of size 64 each for small
#     # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
#     net_arch = [
#         {"pi": [64], "vf": [64]}
#         if net_arch == "tiny"
#         else {"pi": [64, 64], "vf": [64, 64]}
#     ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "gamma": gamma,
        # "buffer_size": buffer_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }
