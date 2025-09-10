import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    stats: dict


class Backtester:
    def __init__(self, data: pd.DataFrame, commission: float = 0.004, slippage: float = 0.0005):
        """
        :param data: DataFrame с колонками ['date', 'open', 'high', 'low', 'close', 'volume']
        :param commission: комиссия на сделку (например, 0.001 = 0.1%)
        :param slippage: проскальзывание (например, 0.0005 = 0.05%)
        """
        self.data = data.copy().reset_index(drop=True)
        self.commission = commission
        self.slippage = slippage

    def run(self, signals: pd.Series, initial_capital: float = 1_000_000) -> BacktestResult:
        """
        :param signals: Series с позицией [-1, 0, 1] на каждый бар (short, flat, long)
        :param initial_capital: стартовый капитал
        """
        df = self.data.copy()
        df["signal"] = signals.shift(1).fillna(0)  # позиция действует со следующего бара

        # Доходности рынка
        df["ret"] = df["close"].pct_change().fillna(0)

        # Доходности стратегии
        df["strategy_ret"] = df["signal"] * df["ret"]

        # Издержки при смене позиции
        df["trade_change"] = df["signal"].diff().abs().fillna(0)
        df["costs"] = df["trade_change"] * (self.commission + self.slippage)
        df["net_ret"] = df["strategy_ret"] - df["costs"]

        # Кривая капитала
        df["equity"] = (1 + df["net_ret"]).cumprod() * initial_capital

        # Сделки
        trades = []
        position = 0
        entry_price = None
        entry_date = None

        for i, row in df.iterrows():
            if position == 0 and row.signal != 0:
                # открываем сделку
                position = row.signal
                entry_price = row.close
                entry_date = row.date
            elif position != 0 and row.signal != position:
                # закрываем сделку
                exit_price = row.close
                pnl = (exit_price / entry_price - 1) * position
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": row.date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl
                })
                position = row.signal
                entry_price = row.close
                entry_date = row.date if position != 0 else None

        trades = pd.DataFrame(trades)

        # Метрики
        stats = self._calc_stats(df["net_ret"], trades, initial_capital)

        return BacktestResult(df["equity"], trades, stats)

    def _calc_stats(self, returns: pd.Series, trades: pd.DataFrame, initial_capital: float):
        equity = (1 + returns).cumprod()
        cagr = equity.iloc[-1]**(252/len(equity)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = returns.mean()/returns.std() * np.sqrt(252) if vol > 0 else 0
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = returns.mean()/downside if downside > 0 else 0
        max_dd = ((equity / equity.cummax()) - 1).min()
        winrate = (trades["pnl"] > 0).mean() if not trades.empty else 0

        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDD": max_dd,
            "Winrate": winrate,
            "Final Equity": equity.iloc[-1] * initial_capital
        }