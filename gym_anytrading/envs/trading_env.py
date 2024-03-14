from time import time
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym


class Actions(Enum):
    """Defines possible actions in the trading environment.

    Attributes:
        Sell: Represents a sell action.
        Buy: Represents a buy action.
        Hold: Represents a hold (no action) decision.
    """
    Sell = 0
    Buy = 1
    Hold = 2


class Positions(Enum):
    """Defines possible positions in the trading environment.

    Attributes:
        Short: Represents a short position.
        Long: Represents a long position.

    Methods:
        opposite: Returns the opposite trading position.
    """
    Short = 0
    Long = 1

    def opposite(self):
        """Returns the opposite trading position.

        Returns:
            Positions: The opposite trading position.
        """
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):
    """A trading environment for reinforcement learning based on gym.Env.

    This environment simulates trading activities with a focus on buying, selling, and holding actions based on provided market data.

    Attributes:
        metadata (dict): Rendering and FPS metadata for the environment.
        df (DataFrame): The dataset containing market data.
        window_size (int): The size of the observation window.
        render_mode (str): The mode used for rendering the environment. Defaults to None.

    Args:
        df (DataFrame): The input dataset containing market data.
        window_size (int): The size of the window for generating observations.
        render_mode (str, optional): The mode for rendering. Defaults to None.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, render_mode=None):
        """Initializes the TradingEnv with a dataset, window size, and render mode."""
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Configuration options for the environment reset.

        Returns:
            observation (object): The initial observation for the environment.
            info (dict): Diagnostic information useful for debugging.
        """
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}



        # Initialize metrics
        self.max_balance = 0
        self.max_drawdown = 0
        self.total_trades = 0
        self.winning_trades = 0


        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        """Executes a step in the environment given an action.

        Args:
            action (int): The action to be executed.

        Returns:
            observation (object): The agent's observation of the current environment.
            reward (float): Amount of reward returned after previous action.
            done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
            truncated (bool): Whether the episode was truncated before reaching the natural terminal state.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = 0
        trade_made = False  # Flag to indicate if a trade was made
        previous_balance = self._total_profit  # Capture the balance before the trade to calculate drawdown and win/loss

        if action != Actions.Hold.value:
            step_reward = self._calculate_reward(action)
            self._total_reward += step_reward

            self._update_profit(action)

            if (
                    (action == Actions.Buy.value and self._position == Positions.Short) or
                    (action == Actions.Sell.value and self._position == Positions.Long)
            ):
                trade_made = True  # A trade was executed
                self._position = self._position.opposite()
                self._last_trade_tick = self._current_tick

        # Update financial metrics after the trade
        if trade_made:
            # Calculate the change in balance to determine if the trade was winning or losing
            change_in_balance = self._total_profit - previous_balance
            self._update_trade_metrics(change_in_balance)

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)
        
        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _update_trade_metrics(self, change_in_balance):
        # Update maximum drawdown
        self.max_balance = max(self.max_balance, self._total_profit)
        current_drawdown = (self.max_balance - self._total_profit) / self.max_balance if self.max_balance != 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Update win/loss count
        self.total_trades += 1
        if change_in_balance > 0:
            self.winning_trades += 1

    # def _get_info(self):
    #     return dict(
    #         total_reward=self._total_reward,
    #         total_profit=self._total_profit,
    #         position=self._position
    #     )
    def _get_info(self):
        return {
            'total_reward': self._total_reward,
            'total_profit': self._total_profit,
            # 'position': self._position,
            'max_balance': self.max_balance,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
        }



    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
