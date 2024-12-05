import logging
import random
import numpy as np

N = 51


class Auction:
    def __init__(self, min_price: float, max_price: float, log: logging.Logger):
        self.feasible_prices = np.linspace(min_price, max_price, num=N)
        self.log = log
        self.count = 0
        self.__gen_winning_curve()

    def prices(self):
        return 0.5 * (self.feasible_prices[1:] + self.feasible_prices[:-1])

    def net_revenue(self):
        p = self.prices()
        return p.max() - p

    def expectations(self):
        nr = self.net_revenue()
        return np.multiply(nr, self.curve)

    def optimal_price(self):
        op = self.prices()[np.argmax(self.expectations())]
        self.log.debug(f'Optimal Price: {op}')
        return op

    def optimal_price_until(self, price: float) -> float:
        pp = self.prices()
        max_val = 0.0
        best_price = 0.0
        for i, _ in enumerate(pp):
            if price - pp[i] <= 0.0:
                break
            val = (price - pp[i]) * self.curve[i]
            if max_val <= val:
                max_val = val
                best_price = pp[i]
        return best_price

    def step(self, price: float) -> bool:
        self.count += 1
        if self.count % 1000 == 0:
            self.__gen_winning_curve()
        # self.log.debug(f'{price=}')
        return random.uniform(0, 1) <= self.__win_probability(price)

    def __gen_winning_curve(self):
        self.curve = 1 - np.random.default_rng().exponential(scale=1, size=N-1)
        self.curve.sort()
        self.curve = (self.curve - self.curve[0]) / (self.curve[N - 2] - self.curve[0])

    def __win_probability(self, price: float) -> float:
        for i in range(N - 1):
            if self.feasible_prices[i] <= price <= self.feasible_prices[i + 1]:
                return float(self.curve[i])

        self.log.debug(f"unfeasible price: {price} [{self.feasible_prices[0], self.feasible_prices[N-1]}]")
        # Return a default probability (like 0.0) if price is out of range
        return 0.0