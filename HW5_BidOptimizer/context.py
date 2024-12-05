import json
import random

import pandas as pd


class Context:
    def __init__(self, d: dict):
        self.dc = d['dc']
        self.ad_format = d['ad_format']
        self.pub_id = d['pub_id']
        self.bundle_id = d['bundle_id']
        self.tag_id = d['tag_id']
        self.cc = d['cc']
        self.context_hash = d['context_hash']
        self.current_index = 0
        self.df = None
        self.load_data()
        self.min_price = self.__get_min_price()
        self.max_price = self.__get_max_price()
        self.log = None

    def to_string(self) -> str:
        return (f"context:\n\tdc:{self.dc}\n\tad_format:{self.ad_format}\n\tpub_id:{self.pub_id}\n\t"
                f"bundle_id:{self.bundle_id}\n\ttag_id:{self.tag_id}\n\tcc:{self.cc}\n\tctx_hash:{self.context_hash}")

    def load_data(self):
        data_file_name = ('data/' + self.dc +
                          '_' + self.ad_format +
                          '_' + self.bundle_id +
                          '_' + self.tag_id +
                          '_' + self.cc)
        if self.pub_id != "0":
            data_file_name += '_' + self.pub_id
        data_file_name = data_file_name.replace(".", "-")
        data_file_name += '.csv'
        self.df = pd.read_csv(data_file_name)
        self.df = self.df.sort_values(by=['timestamp'], ascending=True).reset_index()

    def gen_floor_price(self, price: float) -> float:
        # generate between in the first quarter of the range [min_price, price]
        d_quarter = (price - self.min_price) / 4.0
        return random.uniform(self.min_price, self.min_price + d_quarter)

    def __get_min_price(self) -> float:
        if self.context_hash not in buckets.keys():
            raise Exception("get_floor_price: unknown context")
        return buckets[self.context_hash][0]

    def __get_max_price(self) -> float:
        if self.context_hash not in buckets.keys():
            raise Exception("get_floor_price: unknown context")
        return buckets[self.context_hash][1]

    def get_price_in_time(self):
        if self.current_index >= self.df.shape[0]:
            raise Exception("get_price_in_time: out of data")
        ret = (self.df.loc[self.current_index, 'pn_bid_price'], self.df.loc[self.current_index, 'timestamp'])
        self.current_index += 1
        return ret


class Contexts:
    def __init__(self):
        self.contexts = []

    def add(self, c: Context):
        self.contexts.append(c)


banner_contexts = Contexts()
native_contexts = Contexts()
video_contexts = Contexts()
buckets = dict()


def read_contexts():
    f = open('data/contexts.json')
    data = json.load(f)
    f.close()

    global banner_contexts
    global native_contexts
    global video_contexts

    for d in data['contexts']:
        if d['ad_format'] == 'banner':
            banner_contexts.add(Context(d))
        if d['ad_format'] == 'native':
            native_contexts.add(Context(d))
        if d['ad_format'] == 'video':
            video_contexts.add(Context(d))


def read_buckets():
    f = open('data/buckets.json')
    data = json.load(f)
    f.close()

    global buckets

    for d in data:
        buckets[d['context_hash']] = d['range']
