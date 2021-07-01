# Numpy Python module is used in this file.
#   Licence: BSD-3-Clause License
#   Link: https://github.com/numpy/numpy

# Pandas Python module is used in this file.
#   Licence: BSD-3-Clause License
#   Link: https://github.com/pandas-dev/pandas

# Matplotlib Python module is used in this file.
#   Licence: BSD compatible
#   Link: https://github.com/matplotlib/matplotlib

from IPython.display import clear_output
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._exploration_env import ExplorationEnvironment
from typing import Union, Tuple


class ExplorationAgent:
    """
    Implementation of an agent acting in accordance with the exploration strategy

    Parameters
    ----------
    env: sberpm.ml.reinforcement_learning._exploration_env.ExplorationEnvironment
            Environment object.

    discount: float, default=0.9
        A parameter that determines the agent's attention on immediate and distant rewards.

    alpha: float, default=0.9
        Learning rate for updating the agent.

    epsilon: float, default=0.5
        The probability of performing a random action instead of the best action.

    eps_scaling: float, default=0.9992
        Multiplier for decreasing epsilon at each iteration of agent training.

    with_prob: bool, default=False
         If True, the agent's random action will be performed
         with the corresponding probabilities from the initial data.
         Uniform distribution will be used otherwise.
    """

    def __init__(self, env: ExplorationEnvironment,
                 discount: float = .9,
                 alpha: float = .09,
                 epsilon: float = 0.5,
                 eps_scaling: float = 0.9992,
                 with_prob: bool = False):

        self.qvalues = defaultdict(lambda: defaultdict(lambda: 0.))
        self.discount = discount
        self.alpha = alpha
        self.default_epsilon = epsilon
        self.epsilon = epsilon
        self.with_prob = with_prob
        self.env = env
        self.eps_scaling = eps_scaling

    def _get_qvalue(self, state: int, action: int) -> float:
        """
        Return Q-value of state-action.

        Parameters
        ----------
        state: int
        action: int

        Returns
        -------
        qvalue: float
        """

        return self.qvalues[state][action]

    def _set_qvalue(self, state: int, action: int, value: float) -> None:
        """
        Set state-action Q-value according to Q-learning policy.

        Parameters
        ----------
        state: int
        action: int
        value: float
        """
        self.qvalues[state][action] = value

    def _get_action_value(self, state: int) -> float:
        """
        Calculates the best action value for a given state.

        Parameters
        ----------
        state: int

        Returns
        -------
        value: float
        """
        possible_actions = self.env.legal_actions[state]
        if len(possible_actions) == 0:
            return 0.0
        action_values = list(map(lambda action: self._get_qvalue(state, action), possible_actions))
        value = max(action_values)

        return value

    def _update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Updates agent's Q-values according to update rule.

        Parameters
        ----------
        state: int
        action: int
        reward: float
        next_state: int
        """
        gamma = self.discount
        learning_rate = self.alpha
        new_qvalue = (1 - learning_rate) * self._get_qvalue(state, action) + learning_rate * (
                reward + gamma * self._get_action_value(next_state))

        self._set_qvalue(state, action, new_qvalue)

    def _get_best_action(self, state: int) -> Union[int, None]:
        """
        Returns the best action for a given state based on their Q-values.

        Parameters
        ----------
        state: int

        Returns
        -------
        best_action: int or None
        """
        possible_actions = self.env.legal_actions[state]
        if len(possible_actions) == 0:
            return None
        qvalues = list(map(lambda action: self._get_qvalue(state, action), possible_actions))
        best_action = possible_actions[np.argmax(qvalues)]

        return best_action

    def _get_action(self, state: int) -> Union[int, None]:
        """
        Takes action in accordance with current policy.

        Parameters
        ----------
        state: int

        Returns
        -------
        chosen_action: int or None
        """
        possible_actions = self.env.legal_actions[state]
        if len(possible_actions) == 0:
            return None
        epsilon = self.epsilon
        prob_random = np.random.uniform(0, 1)

        if prob_random < epsilon:
            if self.with_prob:
                chosen_action = np.random.choice(possible_actions, p=self.env.state_probs[state])
            else:
                chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self._get_best_action(state)

        return chosen_action

    def play_session(self, t_max: int = 10 ** 10) -> Tuple[Tuple[int],
                                                           float,
                                                           float,
                                                           int,
                                                           int]:
        """
        Performs agent training prior to staying in final activity.

        Parameters
        ----------
        t_max: int, default=10 ** 10

        Returns
        -------
        resulting_trace,
        session_duration,
        session_reward,
        n_cycles: int
        presence_in_trace: int (0 or 1)
            1 if trace is present in the real data.
        """
        summary_session_reward = 0.0
        s = self.env.reset()

        for t in range(t_max):
            a = self._get_action(s)
            next_s, r, done = self.env.step(a)
            self._update(s, a, r, next_s)
            s = next_s
            summary_session_reward += r
            if done:
                break

        return tuple(self.env.current_trace), \
               self.env.trace_replay_time, \
               summary_session_reward, \
               self.env.cycle_count, \
               self.env.presence_check

    def fit(self, n_iter: int = 1000, verbose: bool = True) -> pd.DataFrame:
        """
        Trains the agent for a given number of iterations and forms a
        dataframe containing the result of the agent's work.

        Parameters
        ----------
        n_iter: int, default=1000
            Number of iterations.

        verbose: bool, default=True
            If True, plots the reward graph representing the learning process.

        Returns
        -------
        reconstruction_result: pd.DataFrame
        """
        session_rewards = []
        reconstruction_result = pd.DataFrame(columns=['trace', 'duration', 'reward',
                                                      'num_of_cycles', 'presence_in_data'])
        plot_percent = int(n_iter * 0.05)
        mean_reward = []
        iterations = []
        for i in range(n_iter):
            resulting_trace, session_duration, session_reward, n_cycles, presence_in_trace = self.play_session()
            session_rewards.append(session_reward)
            reconstruction_result.loc[len(reconstruction_result)] = [resulting_trace, session_duration, session_reward,
                                                                     n_cycles, presence_in_trace]
            self.epsilon *= self.eps_scaling
            if verbose:
                if i % plot_percent == 0 and i != 0:
                    clear_output(True)
                    mean_reward.append(np.mean(session_rewards[-plot_percent:]))
                    iterations.append(i)
                    plt.title('eps = {:e}, mean reward = {:.1f}'.format(self.epsilon, mean_reward[-1]))
                    plt.plot(iterations, mean_reward)
                    plt.xlabel("Iterations")
                    plt.ylabel("Mean rewards")
                    plt.show()
        return reconstruction_result

    def reset(self) -> None:
        """
        Reset epsilon and Q-values.
        """

        self.epsilon = self.default_epsilon
        self.qvalues = defaultdict(lambda: defaultdict(lambda: 0.))
