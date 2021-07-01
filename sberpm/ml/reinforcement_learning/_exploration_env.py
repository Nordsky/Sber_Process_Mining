# Numpy Python module is used in this file.
#   Licence: BSD-3-Clause License
#   Link: https://github.com/numpy/numpy

# Pandas Python module is used in this file.
#   Licence: BSD-3-Clause License
#   Link: https://github.com/pandas-dev/pandas

from ...metrics import IdMetric
import pandas as pd
import numpy as np

from ..._holder import DataHolder
from typing import Dict, Tuple, Optional, List


class ExplorationEnvironment:
    """
    Class that constructs the environment for the exploration strategy based on the initial data.

    Parameters
    ----------
    data_holder: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns (id, activities, timestamps, etc.).

    clear_start_outliers: bool, default=False
        If True, starting activities with low probabilities will be removed
        to reduce the movement of the agent in space.
    """

    def __init__(self, data_holder: DataHolder, clear_start_outliers: bool = False):
        data_holder.check_or_calc_duration()
        self.time_unit = 60
        self.supp_data = self._add_end_event(data_holder)
        self.start_prob = self._start_prob(data_holder, clear_start_outliers)
        self.node_node_prob = self._get_node_node_cond_prob(data_holder)
        self.edge_duration = self._get_mean_edge_duration(data_holder)
        self.states, self.reversed_dict, self.legal_actions, self.state_probs = self._get_state_action_dict(
            self.node_node_prob)
        self.trace_duration = self._calculate_trace_duration(data_holder)
        self.real_traces = list(data_holder.get_grouped_data(data_holder.activity_column)[data_holder.activity_column])
        self.best_result = {}
        self.reward_design = {}
        self.current_trace = []
        self.presence_check = 0
        self.key_states = None
        self.rewards = None
        self.trace_replay_time = None
        self.cycle_count = None
        self.time_scaling = None

    def _add_end_event(self, data_holder: DataHolder) -> pd.DataFrame:
        """
        Adds 'end_event' (and its zero time duration) to the traces in the event log.

        Parameters
        ----------
        data_holder: sberpm.DataHolder

        Returns
        -------
        data: pandas.DataFrame
            Modified log data with 'end_event',
            columns: [activity column (str), time duration column (float (minutes)].
        """
        supp_data = data_holder.get_grouped_data(data_holder.activity_column, data_holder.duration_column)
        supp_data['act_end'] = [tuple(['end'])] * supp_data.shape[0]
        supp_data['time_end'] = [tuple([0])] * supp_data.shape[0]
        supp_data[data_holder.activity_column] = supp_data[data_holder.activity_column] + supp_data['act_end']
        supp_data[data_holder.duration_column] = supp_data[data_holder.duration_column] + supp_data['time_end']

        supp_data = supp_data[[data_holder.id_column, data_holder.activity_column, data_holder.duration_column]] \
            .apply(pd.Series.explode).reset_index(drop=True)
        supp_data[data_holder.duration_column] = supp_data[data_holder.duration_column].fillna(0)

        supp_data[data_holder.duration_column] = supp_data[data_holder.duration_column] / self.time_unit

        return supp_data

    def _get_mean_edge_duration(self, data_holder: DataHolder) -> Dict[Tuple[str, str], float]:
        """
        Calculate approximate time durations of edges.

        Parameters
        ----------
        data_holder: sberpm.DataHolder

        Returns
        -------
        edges_duration: dict of {(str, str): float}
            Key: edge, value: array of probable time durations.
        """

        edges_duration = {}
        df = pd.DataFrame({'edge': zip(self.supp_data[data_holder.activity_column],
                                       self.supp_data[data_holder.activity_column].shift(-1)),
                           'duration': self.supp_data[data_holder.duration_column]})
        id_mask = self.supp_data[data_holder.id_column] == self.supp_data[data_holder.id_column].shift(-1)
        df = df[id_mask]
        edges_duration_array = df.groupby('edge').agg({'duration': tuple})
        for edge, duration_array in zip(edges_duration_array.index, edges_duration_array.values):
            edges_duration[edge] = np.mean(duration_array[0])

        return edges_duration

    @staticmethod
    def _start_prob(data_holder: DataHolder, clear_start_outliers: bool) -> pd.Series:
        """
        Calculates the probabilities for activities to be a start activity in a trace.

        Parameters
        ----------
        data_holder: sberpm.DataHolder

        clear_start_outliers: bool
            If true, remove start activities that occur
            as start activities in less than 5% of traces.

        Returns
        -------
        probs: pd.Series
            Index: activity_name, value: probability.
        """

        id_mask = data_holder.data[data_holder.id_column] != data_holder.data[data_holder.id_column].shift(1)
        activities = data_holder.data[data_holder.activity_column][id_mask]
        probs = activities.value_counts(normalize=True)
        if clear_start_outliers:
            probs = probs[probs > 0.05]

        return probs

    def _get_node_node_cond_prob(self, data_holder: DataHolder) -> Dict[str, pd.Series]:
        """
        Gets the conditional probabilities of the second nodes for each first node in a pair (edge).

        Parameters
        ----------
        data_holder: sberpm.DataHolder

        Returns
        -------
        probs: dict of {str, pd.Series}
            Key: first node, value: pandas.Series: Index: second node, Value: its conditional probability.
        """
        df = pd.DataFrame({'node_1': self.supp_data[data_holder.activity_column],
                           'node_2': self.supp_data[data_holder.activity_column].shift(-1)})
        id_mask = self.supp_data[data_holder.id_column] == self.supp_data[data_holder.id_column].shift(-1)
        df = df[id_mask]
        multi_probs = df.groupby('node_1')['node_2'].value_counts(normalize=True)

        return self._to_prob(multi_probs)

    @staticmethod
    def _to_prob(multiindex_prob_series: pd.Series) -> Dict[str, pd.Series]:
        """
        Converts multiindex Series to dict of conditional probabilities.

        Parameters
        ----------
        multiindex_prob_series: pandas.Series
            Multiindex: (obj_1, obj_2), values: conditional probability of object_2 for given object_1.

        Returns
        -------
        second_object_cond_probs: dict of (obj_1, pandas.Series)
            Key: first object (node: str, or edge: tuple(str, str)).
            Value: Series, Index: object_2 (node: str), values: conditional probabilities: float.
        """
        second_object_cond_probs = {}
        for obj1, obj2_probs in multiindex_prob_series.groupby(level=0):
            obj2_probs = obj2_probs.droplevel(0)
            obj2_probs = obj2_probs[obj2_probs > 0.05]
            clean_probs = obj2_probs / obj2_probs.sum()
            second_object_cond_probs[obj1] = clean_probs

        return second_object_cond_probs

    @staticmethod
    def _get_state_action_dict(transition_dict: Dict[str, pd.Series]) -> Tuple[Dict[str, int],
                                                                               Dict[int, str],
                                                                               Dict[int, Tuple[int]],
                                                                               Dict[int, Tuple[float]]]:
        """
        Encodes each activity in node_node_prob dict and calculates possible actions for each activity,
        dictionary for reverse encoding and state-action probabilities.

        Parameters
        ----------
        transition_dict: dict
            Key: node_1, value: pd.Series: (Index: node_2, value: probability).

        Returns
        -------
        states: dict
            Encoding of states.

        reversed_dict: dict
            Reversed encoding of states.

        legal_actions: dict
            List of next possible encoded states.

        state_action_prob: dict
            List of probabilities of next possible encoded states.
        """
        states = {state: num for num, state in enumerate(transition_dict.keys())}
        states['end'] = len(transition_dict.keys())
        reversed_dict = {v: k for k, v in states.items()}
        legal_actions = {states[node]: tuple(states[n] for n in transition_dict[node].index) for node in states if
                         node in transition_dict}
        state_action_probs = {states[node]: tuple(transition_dict[node].values) for node in states if
                              node in transition_dict}
        legal_actions[states['end']] = []

        return states, reversed_dict, legal_actions, state_action_probs

    def define_rewards(self, reward_design: Dict[str, float],
                       key_states: Optional[List[str]] = None,
                       time_scaling: bool = True):
        """
        Defines reward design and states with the increased reward.

        Parameters
        ----------
        reward_design: dict

        key_states: list, default=None
            Transition to these states will be rewarded additionally.

        time_scaling: bool, default=True
            If True, duration reward will be calculated using the formula:
                final_duration_reward=duration_reward*(mean_trace_duration-current_trace_duration)
        """
        self.reward_design = reward_design
        self.key_states = key_states
        self.time_scaling = time_scaling

    def _calculate_trace_duration(self, data_holder: DataHolder) -> Dict[str, float]:
        """
        Calculates mean trace duration for each starting activity.

        Parameters
        ----------
        data_holder: sberpm.DataHolder

        Returns
        -------
        trace_duration: dict
        """
        trace_duration = {}
        im = IdMetric(data_holder, 'm')
        metrics = im.calc_metrics('trace', 'total_duration')

        for state in self.start_prob.index:
            trace_duration[state] = metrics[metrics['trace'].apply(lambda x: x[0] == state)]['total_duration'].mean()

        return trace_duration

    def reset(self, with_prob: bool = False) -> int:
        """
        Resets the environment to its original state by selecting
        the next starting activity and clearing the internal variables.

        Parameters
        ----------
        with_prob: bool, default=False
            If True, starting activities will be chosen according to their probabilities.
            Otherwise, with uniform distribution.

        Returns
        -------
        state: int
            Encoded starting state.
        """
        self.presence_check = 0
        self.trace_replay_time = 0
        self.current_trace = []
        self.cycle_count = 0

        if with_prob:
            start_state = np.random.choice(self.start_prob.index, p=self.start_prob.values)
        else:
            start_state = np.random.choice(self.start_prob.index)
        self.current_trace.append(start_state)

        return self.states[start_state]

    def _calculate_final_rewards(self) -> float:
        """
        Calculates the final rewards after the session ends.

        Returns
        -------
        final_reward: float
        """
        replay_time = self.trace_replay_time
        final_reward = self.reward_design['finish_reward']

        if self.time_scaling:
            if replay_time < self.trace_duration[self.current_trace[0]]:
                final_reward += self.reward_design['duration_reward'] \
                                * (self.trace_duration[self.current_trace[0]] - replay_time)
        else:
            if self.current_trace[0] in self.best_result:
                if replay_time < self.best_result[self.current_trace[0]]:
                    final_reward += 300
                    self.best_result[self.current_trace[0]] = replay_time
            if replay_time < self.trace_duration[self.current_trace[0]]:
                final_reward += self.reward_design['duration_reward']
                self.best_result[self.current_trace[0]] = replay_time

        if tuple(self.current_trace[:-1]) in self.real_traces:
            final_reward += self.reward_design['presence_reward']
            self.presence_check = 1

        if self.cycle_count == 0:
            final_reward += self.reward_design['final_cycle_reward']

        return final_reward

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Takes a step inside the environment depending on the action performed by the agent
        and calculates intermediate rewards of the step.

        Parameters
        ----------
        action: int

        Returns
        -------
        next_state: int
            Encoded next state.

        reward: float

        done: bool
        """
        done = False
        step_reward = 0
        node_state = self.reversed_dict[action]

        if node_state in self.current_trace:
            self.cycle_count += 1
            step_reward -= self.reward_design['cycle_penalty']

        if self.key_states is not None:
            if node_state in self.key_states:
                step_reward += self.reward_design['increased_reward']

        edge = (self.current_trace[-1], node_state)
        self.trace_replay_time += self.edge_duration[edge]
        self.current_trace.append(node_state)
        step_reward += self.reward_design['default_reward']

        if node_state == 'end':
            done = True
            step_reward += self._calculate_final_rewards()
        next_state = action

        return next_state, step_reward, done
