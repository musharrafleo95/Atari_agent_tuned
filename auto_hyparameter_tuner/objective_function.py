# RL imports
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env


# Autoparameter tuning imports
import optuna

# file imports
from auto_hyparameter_tuner.evaluation_callback import TrialEvalCallback
from auto_hyparameter_tuner.hyper_parameter_sampler import sample_dqn_params

class Objective_function:
    def __init__(
            self,
            default_hyperparams,
            sampler_function,
            rl_enviroment,
            env_name,
            num_eval_env,
            num_eval_episodes,
            eval_freq,
            time_budget,
            model,
            trial_call_back,
            wrapper_class=None,
            wrapper_kwargs=None,
            seed=0,
            eval_deterministic=True
    ):
        self.default_hyperparams = default_hyperparams
        self.sampler = sampler_function
        self.rl_env = rl_enviroment
        self.env_name = env_name
        self.num_eval_env = num_eval_env
        self.num_eval_episodes = num_eval_episodes
        self.eval_freq = eval_freq
        self.wrapper_class = wrapper_class
        self.wrapper_kwargs = wrapper_kwargs
        self.time_budget = time_budget
        self.model = model
        self.trial_call_back = trial_call_back
        self.seed = seed
        self.eval_deterministic = eval_deterministic

    def objective(self, trial: optuna.Trial):

        """
        Objective function using by Optuna to evaluate
        one configuration (i.e., one set of hyperparameters).

        Given a trial object, it will sample hyperparameters,
        evaluate it and report the result (mean episodic reward after training)

        :param trial: Optuna trial object
        :return: Mean episodic reward after training
        """

        kwargs = self.default_hyperparams.copy()

        # 1. Sample hyperparameters and update the keyword arguments
        kwargs.update(self.sampler(trial))
        kwargs.update({"env": self.rl_env.reset(seed=self.seed)})
        # Create the RL model
        model = self.model(**kwargs)

        # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
        eval_envs = make_vec_env(
            self.env_name,
            n_envs=self.num_eval_env,
            wrapper_class=self.wrapper_class,
            wrapper_kwargs=self.wrapper_kwargs
        )
        # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
        # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
        # TrialEvalCallback signature:
        # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
        eval_callback = self.trial_call_back(
            eval_envs,
            trial,
            self.num_eval_episodes,
            self.eval_freq,
            deterministic=self.eval_deterministic
        )

        ### END OF YOUR CODE

        nan_encountered = False
        try:
            # Train the model
            model.learn(self.time_budget, callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            # Free memory
            model.env.close()
            eval_envs.close()

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

# def objective(trial: optuna.Trial) -> float:
#     """
#     Objective function using by Optuna to evaluate
#     one configuration (i.e., one set of hyperparameters).
#
#     Given a trial object, it will sample hyperparameters,
#     evaluate it and report the result (mean episodic reward after training)
#
#     :param trial: Optuna trial object
#     :return: Mean episodic reward after training
#     """
#
#     kwargs = DEFAULT_HYPERPARAMS.copy()
#
#     # 1. Sample hyperparameters and update the keyword arguments
#     kwargs.update(sample_dqn_params(trial))
#     kwargs.update({"env": wrapped_env})
#     # Create the RL model
#     model = DQN(**kwargs)
#
#     # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
#     eval_envs = make_vec_env(ENV_NAME,
#                          n_envs=NUM_EVAL_ENV,
#                          wrapper_class=ConcateObs,
#                          wrapper_kwargs=dict(
#                              k=NUM_FRAMES,
#                              normalize_frame=False
#                          )
#                         )
#     # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
#     # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
#     # TrialEvalCallback signature:
#     # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
#     eval_callback = TrialEvalCallback(eval_envs,
#                                       trial,
#                                       NUM_EVAL_EPISODES,
#                                       EVAL_FREQ,
#                                       deterministic=True)
#
#     ### END OF YOUR CODE
#
#     nan_encountered = False
#     try:
#         # Train the model
#         model.learn(BUDGET, callback=eval_callback)
#     except AssertionError as e:
#         # Sometimes, random hyperparams can generate NaN
#         print(e)
#         nan_encountered = True
#     finally:
#         # Free memory
#         model.env.close()
#         eval_envs.close()
#
#     # Tell the optimizer that the trial failed
#     if nan_encountered:
#         return float("nan")
#
#     if eval_callback.is_pruned:
#         raise optuna.exceptions.TrialPruned()
#
#     return eval_callback.last_mean_reward
