from flask import Flask, request, jsonify
from space_logic import load_spaces
import logging
import threading
import time
import json

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("optimizer-server")

def load_config(config_file: str) -> dict:
    try:
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        logger.debug(f"Loaded configuration: {cfg}")
        return cfg
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_file}: {e}")
        raise

config = load_config("data/config.json")

def load_buckets(buckets_file: str) -> dict:
    try:
        with open(buckets_file, 'r') as f:
            data = json.load(f)
        buckets_map = {d["context_hash"]: d["range"] for d in data}
        logger.debug(f"Loaded buckets map: {buckets_map}")
        return buckets_map
    except Exception as e:
        logger.error(f"Failed to load buckets from {buckets_file}: {e}")
        raise

buckets_map = load_buckets("data/buckets.json")

try:
    spaces_dict = load_spaces(config, buckets_map)
    logger.debug(f"Loaded spaces: {list(spaces_dict.keys())}")
except Exception as e:
    logger.error(f"Failed to load spaces: {e}")
    raise

request_store = {}
EXPLORATION_ROUNDS = 3

def background_learning():
    while True:
        logger.debug("Starting background learning task")
        for sp in spaces_dict.values():
            try:
                sp.learn()
                logger.debug(f"Completed learning for context_hash={sp.context_hash}")
            except Exception as e:
                logger.error(f"Error during learning for context_hash={sp.context_hash}: {e}")
        time.sleep(60)

learning_thread = threading.Thread(target=background_learning, daemon=True)
learning_thread.start()

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    if not data:
        logger.debug("No JSON received in optimize request")
        return jsonify({"error": "Invalid JSON"}), 400

    logger.debug(f"Received optimize request data: {data}")

    req_id = data.get("id", "")
    price = data.get("price", None)
    floor_price = data.get("floor_price", None)
    ctx_hash = data.get("ctx_hash", "default_ctx")  # Assign default if missing
    dc = data.get("data_center", "")
    ad_format = data.get("ext_ad_format", "")
    pub_id = data.get("app_publisher_id", "")
    bundle_id = data.get("bundle_id", "")
    tag_id = data.get("tag_id", "")
    cc = data.get("device_geo_country", "")

    missing_fields = []
    if not req_id:
        missing_fields.append("id")
    if price is None:
        missing_fields.append("price")
    if floor_price is None:
        missing_fields.append("floor_price")

    if missing_fields:
        logger.debug(f"Missing required fields: {', '.join(missing_fields)}")
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    if ctx_hash not in spaces_dict:
        logger.debug(f"Unknown context_hash: {ctx_hash}")
        return jsonify({"error": f"Unknown context {ctx_hash}"}), 400

    if price < floor_price:
        logger.debug(f"Price < floor_price: price={price}, floor_price={floor_price}")
        return jsonify({"optimized_price": 0.0, "status": "error"}), 200

    if req_id not in request_store:
        request_store[req_id] = {
            "price": price,
            "floor_price": floor_price,
            "attempts": 0,
            "impressions": 0,
            "context_hash": ctx_hash,
            "data_center": dc,
            "ad_format": ad_format,
            "pub_id": pub_id,
            "bundle_id": bundle_id,
            "tag_id": tag_id,
            "cc": cc
        }

    state = request_store[req_id]
    state["attempts"] += 1

    candidate_price = price * 0.9
    optimized_price = max(candidate_price, floor_price)

    if state["attempts"] <= EXPLORATION_ROUNDS:
        status = "explored"
    else:
        if state["impressions"] > 0:
            status = "optimized"
        else:
            status = "explored"

    logger.debug(
        f"OPTIMIZE: req_id={req_id}, price={price}, floor_price={floor_price}, "
        f"optimized_price={optimized_price}, status={status}, ctx_hash={ctx_hash}, "
        f"dc={dc}, ad_format={ad_format}, pub_id={pub_id}, bundle_id={bundle_id}, tag_id={tag_id}, cc={cc}"
    )

    return jsonify({"optimized_price": optimized_price, "status": status}), 200

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data:
        logger.debug("No JSON received in feedback request")
        return jsonify({"error": "Invalid JSON"}), 400

    logger.debug(f"Received feedback data: {data}")

    req_id = data.get("id", "")
    price = data.get("price", None)
    impression = data.get("impression", None)

    missing_fields = []
    if not req_id:
        missing_fields.append("id")
    if price is None:
        missing_fields.append("price")
    if impression is None:
        missing_fields.append("impression")

    if missing_fields:
        logger.debug(f"Missing fields in feedback: {', '.join(missing_fields)}")
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    if req_id not in request_store:
        logger.debug(f"FEEDBACK: Unknown req_id={req_id}")
        return jsonify({"ack": False}), 200

    state = request_store[req_id]
    state["impressions"] += 1
    ctx_hash = state["context_hash"]
    sp = spaces_dict[ctx_hash]

    buckets_indices = [lvl.sample_buckets(price) for lvl in sp.levels]

    sp.update_feedback(buckets_indices, impression)

    logger.debug(f"FEEDBACK: req_id={req_id}, price={price}, impression={impression}, total_impressions={state['impressions']}")

    return jsonify({"ack": True}), 200

@app.route('/space', methods=['GET'])
def space_endpoint():
    ctx = request.args.get('ctx', 'default_ctx')  # Assign default if missing
    if ctx not in spaces_dict:
        logger.debug(f"Space endpoint called with unknown ctx: {ctx}")
        return jsonify({}), 200

    sp = spaces_dict[ctx]
    learned_estimation = sp.WC()
    logger.debug(f"Space endpoint response for ctx={ctx}: {learned_estimation}")

    return jsonify(learned_estimation), 200

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
