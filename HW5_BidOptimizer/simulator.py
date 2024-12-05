import logging
from collections import deque
import context
import client
import auction


class Simulator:
    def __init__(self, cln: client.Client, ctx: context.Context, log: logging.Logger):
        self.client = cln
        self.context = ctx
        self.signal = False
        self.log = log
        self.auction = auction.Auction(self.context.min_price, self.context.max_price, log)
        self.d = deque()

    def stop(self):
        self.signal = True

    def save(self, bid_price: float, optimized_price: float):
        true_best_price = self.auction.optimal_price_until(bid_price)
        if len(self.d) > 2:
            self.d.popleft()
        self.d.append((true_best_price, optimized_price, bid_price))

    def run(self):
        while not self.signal:
            bid_response = self.client.send_bid_request()
            if bid_response.status == "error":
                self.log.debug(" ---> send_bid_request(): status == error")
                continue

            if bid_response.optimized_price > bid_response.price_to_bid:
                raise Exception("simulator.run(): optimized price > bid price")
            if bid_response.status != "explored":
                self.save(bid_response.price_to_bid, bid_response.optimized_price)
                continue
            impression = self.auction.step(bid_response.optimized_price)

            if impression:
                res = self.client.send_impression(bid_response.req_id, bid_response.optimized_price, impression)
                if not res:
                    self.log.debug(" ---> send_impression(): ack == false")
