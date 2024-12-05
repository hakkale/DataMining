from threading import Thread
import context
import client
import simulator
from log import get_logger
import animate


def simulate(sim: simulator.Simulator):
    sim.run()


if __name__ == '__main__':
    log = get_logger("client", "/tmp/tail_client.log")
    log.debug("Loading buckets.json")
    context.read_buckets()
    log.debug("Loading contexts.json")
    context.read_contexts()

    context_ = context.banner_contexts.contexts[0]
    context_.log = log
    log.debug(context_.to_string())

    client = client.Client(context=context_, host="127.0.0.1", port=8000, log=log)
    log.debug("Simulator run ...")
    simulator = simulator.Simulator(cln=client, ctx=context_, log=log)
    thread = Thread(target=simulate, args=(simulator,))
    thread.start()
    log.debug("Run animate ...")
    animate.run_animate(host="127.0.0.1", port=8000, context=context_, simulator=simulator, log=log)

    simulator.stop()
    thread.join()