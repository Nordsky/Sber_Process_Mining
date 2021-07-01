# Pandas Python module is used in this file.
#   Licence: BSD-3-Clause License
#   Link: https://github.com/pandas-dev/pandas

from ...metrics import TransitionMetric
from ..reinforcement_learning._exploration_agent import ExplorationAgent
from ..reinforcement_learning._exploration_env import ExplorationEnvironment
from ..reinforcement_learning._exploitation_agent import ExploitationAgent
from ..reinforcement_learning._exploitation_env import ExploitationEnvironment
import pandas as pd

from ..._holder import DataHolder

from typing import Optional, List, Dict, Union


class RLOptimizer:
    """
    Class used to apply reinforcement learning to reconstruct processes or find optimal paths.
    To work with a class, you need to choose one of the two strategies, 'exploration' or 'exploitation',
    on which the result of optimization will depend.
    Descriptions of the parameters of some methods are presented in classes corresponding to each strategy.

    Parameters
    ----------
    data_holder: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    strategy: {'exploration', 'exploitation'}, default='exploitation'
        Choosing an agent training strategy.

        Exploration means exploring the environment with the ability to perform
        more effective, but not previously used actions.

        Exploitation involves the performance of known paths and actions
        in order to find the most effective ones among them.


    env_args: dict, default=None
        Parameters of the environment.

        Parameters for 'exploration' starategy:
            -clear_start_outliers:
                If True, starting activities with low probabilities
                (less than 5%) will be removed to reduce the movement
                of the agent in space.

        Default values:
            'clear_start_outliers': False.


        Parameters for 'exploitation' starategy:
            -shuffle:
                If True, the order of the traces will be changed randomly.

        Default values:
            'shuffle': True.


    agent_args: dict, default=None
        Parameters of the agent.

        Parameters for 'exploration' starategy:
            -discount:
                determines the agent's attention on immediate
                and distant rewards
            -alpha:
                learning rate for updating the agent
            -epsilon:
                probability of performing a random action instead of
                the best action
            -eps_scaling:
                multiplier for decreasing epsilon at each iteration
                of agent training
            -with_prob:
                if True, the agent's random action will be performed
                with the corresponding probabilities from the initial data.
                Uniform distribution will be used otherwise.

        Default values:
            'discount': 0.9,
            'alpha': 0.09,
            'epsilon': 0.5,
            'eps_scaling': 0.9992,
            'with_prob': False.


        Parameters for 'exploitation' starategy:
            -discount:
                determines the agent's attention on immediate
                and distant rewards
            -alpha:
                learning rate for updating the agent

        Default values:
            'discount': 0.9,
            'alpha': 0.1.

    Attributes
    ----------
    dh: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    strategy: str
        Selected strategy.

    reward_design: dict
        Design of rewards used in agent training.

    result: pd.DataFrame
        The result of agent training, which depends on the chosen strategy.

    Examples
    --------
    >>> from sberpm.ml.reinforcement_learning import RLOptimizer
    >>> rl = RLOptimizer(data_holder)
    >>> rl.define_rewards()
    >>> rl.fit()
    >>> rl.get_optimal_paths()
    """

    def __init__(self, data_holder: DataHolder,
                 strategy: str = 'exploitation',
                 env_args: Optional[dict] = None,
                 agent_args: Optional[dict] = None):
        self.dh = data_holder
        self.strategy = strategy
        if self.strategy == 'exploration':
            default_env_args = {'clear_start_outliers': False}
            default_agent_args = {'discount': .9, 'alpha': .09, 'epsilon': 0.5, 'eps_scaling': 0.9992,
                                  'with_prob': False}
        elif self.strategy == 'exploitation':
            default_env_args = {'shuffle': True}
            default_agent_args = {'discount': .9, 'alpha': .1}
        else:
            raise ValueError(f"Strategy should be 'exploration', or 'exploitation' but got '{strategy}' instead.")

        if env_args is None:
            env_args = default_env_args
        if agent_args is None:
            agent_args = default_agent_args

        if self.strategy == 'exploration':
            self.env = ExplorationEnvironment(data_holder=self.dh, **env_args)
            self.agent = ExplorationAgent(env=self.env, **agent_args)
        else:
            self.env = ExploitationEnvironment(data_holder=self.dh, **env_args)
            self.agent = ExploitationAgent(env=self.env, **agent_args)
        self.reward_design = None
        self.result = None

    def define_rewards(self, reward_design: Optional[Dict[str, float]] = None,
                       auto: bool = False,
                       time_scaling: bool = True,
                       key_states: Optional[List[str]] = None):
        """
        Defines reward design for agent training.

        Parameters
        ----------
        reward_design: dict, default=None
            Description of rewards:
                -default_reward: agent's reward for performing an action
                                leading to a transition to the next activity
                -increased_reward: agent's reward for performing an action
                                leading to a transition to a 'key activity'
                -finish_reward: agent's reward for performing an action
                                leading to a transition to a final activity of a trace
                -duration_reward: 1. agent reward for completing the trace in less than the average
                                trace time in the original data (in 'exploration' mode)
                                  2. agent's reward for making a transition that is shorter in duration
                                than the average time of this transition in the initial data (in 'exploitation' mode)
                -cycle_penalty: immediate penalty for looping
                -final_cycle_reward: agent's reward for completing a trace without loops
                -presence_reward: agent's reward for building the trace
                                contained in the initial data (in 'exploration' mode only)

            Default reward design for 'exploration' strategy:
                'default_reward': 15,
                'increased_reward': 0,
                'finish_reward': 100,
                'duration_reward': 0.1,
                'cycle_penalty': 20,
                'final_cycle_reward': 150,
                'presence_reward': 1500.

            Default reward design for 'exploitation' strategy:
                'default_reward': 2,
                'increased_reward': 0,
                'finish_reward': 20,
                'duration_reward': 10,
                'cycle_penalty': 30,
                'final_cycle_reward': 0.

        auto: bool, default=None
            If True, performs automatic selection of rewards base on given data
            (for 'exploitation' strategy only). Overwrites reward_design argument.

        time_scaling: bool, default=True
            If True, duration reward will be calculated using the formula:
            final_duration_reward = duration_reward * (mean_trace_duration - current_trace_duration)
            (for 'exploration' strategy only).

        key_states: list, default=None
            Transition to these states will be rewarded additionally.
        """
        if self.strategy == 'exploration':
            default_reward_design = {'default_reward': 15, 'increased_reward': 0, 'finish_reward': 100,
                                     'duration_reward': 0.1, 'cycle_penalty': 20, 'final_cycle_reward': 150,
                                     'presence_reward': 1500}
        else:
            default_reward_design = {'default_reward': 2, 'increased_reward': 0, 'finish_reward': 20,
                                     'duration_reward': 10, 'cycle_penalty': 30, 'final_cycle_reward': 0}
        if auto:
            if self.strategy == 'exploitation':
                reward_design = self._calculate_reward_design()
            else:
                raise NotImplementedError('Automatic selection of rewards is available only when choosing an '
                                          '"exploitation" strategy.')

        if reward_design is None and self.reward_design is None:
            self.reward_design = default_reward_design
        elif reward_design is not None:
            self.reward_design = reward_design

        if self.strategy == 'exploration':
            self.env.define_rewards(reward_design=self.reward_design, time_scaling=time_scaling, key_states=key_states)
        else:
            self.env.define_rewards(reward_design=self.reward_design, key_states=key_states)

    def _calculate_reward_design(self) -> Dict[str, float]:
        """
        Performs automatic selection of rewards for exploitation environment.

        Returns
        -------
        reward_design: dict
        """
        reward_design = dict()
        reward_design['default_reward'] = 1
        reward_design['duration_reward'] = 2
        reward_design['final_cycle_reward'] = 0
        gd = self.dh.get_grouped_data(self.dh.activity_column)[self.dh.activity_column]
        cm = TransitionMetric(self.dh)
        n_cycles = sum(cm.count() * cm.loop_percent() / 100)
        n_steps = gd.apply(len).median()
        reward_design['cycle_penalty'] = (reward_design['default_reward'] + reward_design['duration_reward']) * -2
        reward_design['finish_reward'] = - reward_design['cycle_penalty'] * n_cycles - \
                                         (reward_design['default_reward'] +
                                          reward_design['duration_reward']) * n_steps

        return reward_design

    def fit(self, n_iter: Optional[int] = None, verbose=True):
        """
        Fits selected agent.

        Parameters
        ----------
        n_iter: int, default=None
            Number of iterations/sessions to perform.
            (=Number of traces to make.)
        
        verbose: bool, default=True
            If True, plots the reward graph representing the learning process.
            Horizontal axis represents a number of completed sessions,
            vertical axis shows mean reward for last
            int(n_iter * 0.05) sessions.

        Returns
        -------
        result: pd.DataFrame
        """
        if self.reward_design is None:
            raise RuntimeError('Call define_rewards() method first.')
        if self.strategy == 'exploration':
            default_n_iter = 10000
        else:
            default_n_iter = len(self.dh.get_grouped_data())
        if n_iter is None:
            n_iter = default_n_iter

        self.result = self.agent.fit(n_iter=n_iter, verbose=verbose)

        return self.result

    def reset(self):
        """
        Reset agent's Q-values and epsilon variables.
        """
        self.agent.reset()

    def get_optimal_paths(self, max_actions: int = 25) -> Union[pd.DataFrame, List[str]]:
        """
        Calculates the resulting traces with the best rewards or the optimal path depending on the chosen strategy.

        Parameters
        ----------
        max_actions: int, default=25
            Maximum number of actions (only for 'exploitation' strategy).

        Returns
        -------
        result: pd.DataFrame (exploration) or list (exploitation).
        """
        if self.result is None:
            raise RuntimeError('Call fit() method first.')
        if self.strategy == 'exploitation':
            return self._get_optimal_exploitation_path(max_actions)
        else:
            return self._get_optimal_exploration_paths()

    def _get_optimal_exploitation_path(self, max_actions: int) -> List[str]:
        """
        Calculates optimal path (sequence of actions) according to agent's Q-values.

        Parameters
        ----------
        max_actions: int
            Maximum number of actions.

        Returns
        -------
        optimal_actions: list of str
            List of actions.
        """
        optimal_actions = []
        action = self.env.initial_state
        optimal_actions.append(action)
        i = 0
        while action != self.env.terminal_state:
            i += 1
            if i == max_actions:
                break

            new_action = self.agent.get_best_action(action)
            optimal_actions.append(new_action)
            action = new_action

        return optimal_actions

    def _get_optimal_exploration_paths(self) -> pd.DataFrame:
        """
        Finds unique traces with the best rewards for each starting activity.

        Returns
        -------
        best_paths: pd.DataFrame
        """
        best_paths = pd.DataFrame()

        for state in self.env.start_prob.index:
            auxiliary_df = self.result[self.result.trace.apply(lambda x: x[0] == state)]
            auxiliary_df = auxiliary_df[auxiliary_df.reward == auxiliary_df.reward.max()]
            auxiliary_df.drop_duplicates(inplace=True)
            best_paths = best_paths.append(auxiliary_df)

        return best_paths
