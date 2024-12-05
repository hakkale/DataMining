import json
import random
from typing import List, Dict
import threading
import lightgbm as lgb
import numpy as np

LAMBDA_MIN = 0.1
LAMBDA_MAX = 1.8
MIN_LEVEL_SIZE = 1
MIN_BUCKET_SIZE = 5
MIN_BUFFER_SIZE = 10

class Bucket:
    def __init__(self, lhs: float, rhs: float, size: int, discount: float):
        self.lhs = lhs
        self.rhs = rhs
        self.alpha = 1.0
        self.beta = 1.0
        self.buffer = []
        self.size = size
        self.discount = discount
        self.pr = 0.5
        self.update_qty = 0

    def update(self, impression: bool):
        self.update_qty += 1
        # If buffer is full, remove the oldest entry
        if len(self.buffer) >= self.size:
            oldest = self.buffer.pop()
            if oldest:
                self.alpha -= 1
            else:
                self.beta -= 1
        # Add the new impression at the front
        self.buffer.insert(0, impression)

        if impression:
            self.alpha += 1
        else:
            self.beta += 1

        # Update pr as a weighted combination (discounted)
        self.pr = self.discount * self.pr + (1 - self.discount) * (self.alpha / (self.alpha + self.beta))

class Level:
    def __init__(self, buckets: List[Bucket]):
        self.buckets = buckets
        self.winning_curve = [0.5 for _ in buckets]

    def sample_buckets(self, price: float) -> int:
        for i, b in enumerate(self.buckets):
            if b.lhs <= price <= b.rhs:
                return i
        return -1

class Space:
    def __init__(self, context_hash: str, levels: List[Level]):
        self.context_hash = context_hash
        self.levels = levels
        self.lock = threading.Lock()
        self.learning_data = {i: [] for i in range(len(levels))}  # Collect data for each level

    def WC(self) -> Dict:
        """Return the LearnedEstimation structure."""
        result = {"level": []}
        for lvl in self.levels:
            level_est = {
                "price": [],
                "pr": []
            }
            for b in lvl.buckets:
                mid_price = b.lhs + (b.rhs - b.lhs) / 2.0
                level_est["price"].append(mid_price)
                level_est["pr"].append(lvl.winning_curve[lvl.buckets.index(b)])
            result["level"].append(level_est)
        return result

    def update_feedback(self, buckets_indices: List[int], impression: bool):
        """Update the buckets based on feedback."""
        with self.lock:
            for i, bID in enumerate(buckets_indices):
                if bID == -1:
                    continue
                self.levels[i].buckets[bID].update(impression)
                self.learning_data[i].append({
                    'pr': self.levels[i].buckets[bID].pr,
                    'price': (self.levels[i].buckets[bID].lhs + self.levels[i].buckets[bID].rhs) / 2.0,
                    'impression': int(impression)
                })

    def learn(self):
        """Update the winning_curve based on learning data."""
        for i, level in enumerate(self.levels):
            data = self.learning_data[i]
            if len(data) < 10:
                continue

            X = np.array([[d['pr'], d['price']] for d in data])
            y = np.array([d['impression'] for d in data])

            train_data = lgb.Dataset(X, label=y)
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1
            }
            model = lgb.train(params, train_data, num_boost_round=10)

            for j, bucket in enumerate(level.buckets):
                mid_price = (bucket.lhs + bucket.rhs) / 2.0
                pred = model.predict(np.array([[bucket.pr, mid_price]]))[0]
                level.winning_curve[j] = pred

            self.learning_data[i] = []

def linspace(level_size: int) -> List[float]:
    """Generate linearly spaced lambda values."""
    lambdas = [0.0] * level_size
    lambdas[0] = LAMBDA_MIN
    if level_size == 1:
        return lambdas
    step = (LAMBDA_MAX - LAMBDA_MIN) / float(level_size - 1)
    for i in range(1, level_size):
        lambdas[i] = lambdas[i - 1] + step
    return lambdas

def generate_bucket_bounds(lambda_param: float, min_price: float, max_price: float, n_buckets: int) -> List[float]:
    """Generate bucket boundaries based on exponential distribution."""
    bounds = []
    for _ in range(n_buckets + 1):
        val = random.expovariate(lambda_param)
        bounds.append(val)
    bounds.sort()

    mi = bounds[0]
    ma = bounds[-1]
    scaled = [((val - mi) / (ma - mi)) * (max_price - min_price) + min_price for val in bounds]
    return scaled

def new_buckets(lambda_param: float, min_price: float, max_price: float, cfg: Dict) -> List[Bucket]:
    """Create new buckets based on lambda parameter."""
    bounds = generate_bucket_bounds(lambda_param, min_price, max_price, cfg['bucket_size'])
    lhs = bounds[:-1]
    rhs = bounds[1:]
    buckets = [Bucket(lhs[i], rhs[i], cfg['buffer_size'], cfg['discount']) for i in range(cfg['bucket_size'])]
    return buckets

def new_levels(min_price: float, max_price: float, cfg: Dict) -> List[Level]:
    """Create new levels based on lambda values."""
    levels = []
    lambdas = linspace(cfg['level_size'])
    for lam in lambdas:
        buckets = new_buckets(lam, min_price, max_price, cfg)
        lvl = Level(buckets)
        levels.append(lvl)
    return levels

def load_buckets(buckets_file: str) -> Dict[str, List[float]]:
    """Load bucket ranges from buckets.json."""
    with open(buckets_file, 'r') as f:
        data = json.load(f)
    buckets_map = {d["context_hash"]: d["range"] for d in data}
    return buckets_map

def load_spaces(cfg: Dict, buckets_map: Dict[str, List[float]]) -> Dict[str, Space]:
    """Load spaces based on config and buckets."""
    space_desc_file = cfg.get("space_desc_file", "")
    if space_desc_file == "":
        raise ValueError("space_desc_file not specified in config.json")

    try:
        with open(space_desc_file, 'r') as f:
            spaces_desc = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load space descriptions from {space_desc_file}: {e}")

    spaces = {}
    for s in spaces_desc:
        context_hash = s["context_hash"]

        if context_hash in buckets_map:
            min_price, max_price = buckets_map[context_hash]
        else:
            min_price = s.get("min_price", 0.0)
            max_price = s.get("max_price", 1.0)

        if cfg['level_size'] < MIN_LEVEL_SIZE:
            raise ValueError(f"Invalid level size: {cfg['level_size']}")
        if cfg['bucket_size'] < MIN_BUCKET_SIZE:
            raise ValueError(f"Invalid bucket size: {cfg['bucket_size']}")
        if cfg['buffer_size'] < MIN_BUFFER_SIZE:
            raise ValueError(f"Invalid buffer size: {cfg['buffer_size']}")
        if not (0.0 <= cfg['discount'] <= 1.0):
            raise ValueError(f"Invalid discount factor: {cfg['discount']}")

        levels = new_levels(min_price, max_price, cfg)
        sp = Space(context_hash, levels)
        spaces[context_hash] = sp
    return spaces
