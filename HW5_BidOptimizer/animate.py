import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
import logging
import context
import simulator

mpl.use('Qt5Agg')


class BidResponse:
    def __init__(self, req_id="", price_to_bid=0.0, optimized_price=0.0, status="error"):
        self.req_id = req_id
        self.price_to_bid = price_to_bid
        self.optimized_price = optimized_price
        self.status = status


class Animate:
    def __init__(self, host: str, port: int, context: context.Context, simulator: simulator.Simulator,
                 log: logging.Logger):
        self.headers = {'Content-Type': 'application/json'}
        self.url = f'http://{host}:{port}'
        self.context = context
        self.dist = self.context.max_price - self.context.min_price
        self.simulator = simulator
        self.log = log
        self.params = {'ctx': self.context.context_hash}

        self.fig, self.ax_true = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        self.fig.tight_layout()

    def quantities(self):
        try:
            response = requests.get(url=self.url + '/space', params=self.params, headers=self.headers)
            if response.status_code != 200:
                self.log.debug(f"Failed to retrieve space data: {response.status_code}")
                return
            resp_json = response.json()
            levels = resp_json.get('level', [])

            self.ax_true.cla()

            self.ax_true.plot(
                self.simulator.auction.prices(),
                self.simulator.auction.curve,
                color='blue',
                label='# True Win Probability'
            )

            self.ax_true.plot(
                self.simulator.auction.prices(),
                self.simulator.auction.net_revenue(),
                color='blue',
                label='# True Net Revenue',
                linestyle='--'
            )

            self.ax_true.plot(
                self.simulator.auction.prices(),
                self.simulator.auction.expectations(),
                color='green',
                label='# True Expectations',
                linestyle='-.'
            )

            optimal_price = self.simulator.auction.optimal_price()
            self.ax_true.scatter(
                [optimal_price],
                [0.0],
                alpha=1.0,
                color='red',
                marker='x',
                s=100,
                label='# Optimal Price'
            )

            self.ax_true.set_title("Optimizer Performance")
            self.ax_true.set_xlabel("Price")
            self.ax_true.set_ylabel("Probability / Revenue")
            self.ax_true.legend()


        except Exception as e:
            self.log.error(f"Error in quantities method: {e}")


def animate_call(i, animate):
    animate.quantities()


def run_animate(host: str, port: int, context: context.Context, simulator: simulator.Simulator, log: logging.Logger):
    animate = Animate(host, port, context, simulator, log)
    animation = FuncAnimation(animate.fig, animate_call, interval=5000, fargs=(animate,))
    plt.tight_layout()
    plt.show(block=True)
