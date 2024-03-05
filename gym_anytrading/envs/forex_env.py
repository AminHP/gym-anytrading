import numpy as np
import pandas as pd
from typing import Tuple, Optional
import pandas_ta as ta
from .trading_env import TradingEnv, Actions, Positions

class ForexEnv(TradingEnv):
    """A trading environment for Forex based on the TradingEnv abstract class.

    This class provides specific implementations for Forex trading, including
    data processing, reward calculation, and profit updates.

    Attributes:
        frame_bound (Tuple[int, int]): A tuple representing the start and end index for the trading frame.
        unit_side (str): A string indicating the side of the unit ('left' or 'right').
        trade_fee (float): The trading fee per trade.
    """
    def __init__(self, df: pd.DataFrame, window_size: int, frame_bound: Tuple[int, int], unit_side: str = 'left', render_mode: Optional[str] = None):
        """Initializes the ForexEnv with the given parameters.

        Args:
            df (pd.DataFrame): The dataset containing price information.
            window_size (int): The size of the moving window that includes the current price and the previous prices.
            frame_bound (Tuple[int, int]): The start and end indices for the trading frame.
            unit_side (str, optional): The side of the unit for profit calculation ('left' or 'right'). Defaults to 'left'.
            render_mode (Optional[str], optional): The mode for rendering the environment. Defaults to None.
        """
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound: Tuple[int, int] = frame_bound
        self.unit_side: str = unit_side.lower()
        super().__init__(df, window_size, render_mode)

        self.trade_fee: float = 0.0003  # unit

    def _process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Processes the raw data to generate features for trading.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the processed prices and signal features as numpy arrays.
        """
        
        self.df = pd.concat([self.df, ta.sma(self.df.Close, length=5)], axis=1, )
        self.df = pd.concat([self.df, ta.sma(self.df.Close, length=14)], axis=1, )
        self.df = pd.concat([self.df, ta.sma(self.df.Close, length=21)], axis=1, )
        self.df = pd.concat([self.df, ta.sma(self.df.Close, length=34)], axis=1, )
        self.df = pd.concat([self.df, ta.sma(self.df.Close, length=50)], axis=1, )
        self.df = pd.concat([self.df, ta.sma(self.df.Close, length=100)], axis=1, )
        self.df = pd.concat([self.df, ta.rsi(self.df.Close)], axis=1, )
        self.df = pd.concat([self.df, ta.adx(self.df.High, self.df.Close, self.df.Low)], axis=1, ) # ADX_14 	DMP_14 	DMN_14
        self.df = pd.concat([self.df, ta.bbands(self.df.Close)], axis=1, )
        self.df = pd.concat([self.df, ta.psar(self.df.High, self.df.Low, self.df.Close)], axis=1, )
        self.df = pd.concat([self.df, ta.aroon(self.df.High, self.df.Low)], axis=1, )
        self.df = pd.concat([self.df, ta.macd(self.df.Close)], axis=1, )


        self.df.dropna(inplace=True)
        self.meaningful_df = pd.DataFrame()


        # Simple Moving Average (SMA) Comparisons
        self.meaningful_df['SMA_5_above_14'] = (self.df['SMA_5'] >= self.df['SMA_14']).astype(int)
        self.meaningful_df['SMA_5_above_21'] = (self.df['SMA_5'] >= self.df['SMA_21']).astype(int)
        self.meaningful_df['SMA_5_above_34'] = (self.df['SMA_5'] >= self.df['SMA_34']).astype(int)
        self.meaningful_df['SMA_5_above_50'] = (self.df['SMA_5'] >= self.df['SMA_50']).astype(int)
        self.meaningful_df['SMA_5_above_100'] = (self.df['SMA_5'] >= self.df['SMA_100']).astype(int)

        self.meaningful_df['SMA_14_above_21'] = (self.df['SMA_14'] >= self.df['SMA_21']).astype(int)
        self.meaningful_df['SMA_14_above_34'] = (self.df['SMA_14'] >= self.df['SMA_34']).astype(int)
        self.meaningful_df['SMA_14_above_50'] = (self.df['SMA_14'] >= self.df['SMA_50']).astype(int)
        self.meaningful_df['SMA_14_above_100'] = (self.df['SMA_14'] >= self.df['SMA_100']).astype(int)

        self.meaningful_df['SMA_21_above_34'] = (self.df['SMA_21'] >= self.df['SMA_34']).astype(int)
        self.meaningful_df['SMA_21_above_50'] = (self.df['SMA_21'] >= self.df['SMA_50']).astype(int)
        self.meaningful_df['SMA_21_above_100'] = (self.df['SMA_21'] >= self.df['SMA_100']).astype(int)

        self.meaningful_df['SMA_34_above_50'] = (self.df['SMA_34'] >= self.df['SMA_50']).astype(int)
        self.meaningful_df['SMA_34_above_100'] = (self.df['SMA_34'] >= self.df['SMA_100']).astype(int)

        self.meaningful_df['SMA_50_above_100'] = (self.meaningful_df['SMA_50'] >= self.meaningful_df['SMA_100']).astype(int)

        # Relative Strength Index (RSI)
        self.meaningful_df['RSI_above_70'] = (self.df['RSI_14'] >= 70).astype(int)
        self.meaningful_df['RSI_above_80'] = (self.df['RSI_14'] >= 80).astype(int)
        self.meaningful_df['RSI_below_30'] = (self.df['RSI_14'] <= 30).astype(int)
        self.meaningful_df['RSI_below_20'] = (self.df['RSI_14'] <= 20).astype(int)

        # Directional Movement Index (ADX)
        self.meaningful_df['ADX_Class'] = self.df['ADX_14'].apply(lambda x: 0 if x < 20 else 1 if x < 25 else 2)
        self.meaningful_df['DMP_bigger_than_DMN'] = (self.df['DMP_14'] > self.df['DMN_14']).astype(int)
        self.meaningful_df['DMP_DMN_diff'] = self.df['DMP_14'] - self.df['DMN_14']

        # Bollinger Bands (BBands)
        self.meaningful_df['BBands_distance_from_lower'] = self.df['Close'] - self.df['BBL_5_2.0']
        self.meaningful_df['BBands_distance_from_upper'] = self.df['BBU_5_2.0'] - self.df['Close']
        self.meaningful_df['BBands_bandwidth'] = self.df['BBB_5_2.0']

        # Parabolic SAR (PSAR)
        self.meaningful_df['PSAR'] = (~self.df['PSARl_0.02_0.2'].isna()).astype(int)

        # Moving Average Convergence Divergence (MACD)
        self.meaningful_df['MACD_signal_above_macd'] = (self.df['MACDs_12_26_9'] >= self.df['MACD_12_26_9']).astype(int)
        self.meaningful_df['MACD_histogram'] = self.df['MACDh_12_26_9']
        self.meaningful_df['MACD_histogram_sign'] = (self.df['MACDh_12_26_9'] >= 0).astype(int)


        prices = self.df.loc[:, 'Close'].to_numpy()

        

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = self.meaningful_df.to_numpy()
        signal_features = np.column_stack((prices, diff, signal_features))
        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action: int) -> float:
        """Calculates the reward based on the given action.

        Args:
            action (int): The action taken by the agent.

        Returns:
            float: The calculated reward for the action.
        """
        step_reward = 0  # pip

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long:
                step_reward += price_diff * 10000

        return step_reward

    def _update_profit(self, action: int):
        """Updates the total profit based on the given action.

        Args:
            action (int): The action taken by the agent.
        """
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * last_trade_price
                    self._total_profit = quantity / (current_price + self.trade_fee) 
                elif self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee) 
            elif self.unit_side == 'right':
                if self._position == Positions.Short:
                    quantity = self._total_profit * last_trade_price
                    self._total_profit = quantity / (current_price + self.trade_fee) 
                elif self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)  

    def max_possible_profit(self) -> float:
        """Calculates the maximum possible profit for the given frame.

        Returns:
            float: The maximum possible profit.
        """
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit
