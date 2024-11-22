import logging
# import atexit
import json
import logging.config


# Init logging
def init_logging(config_path):
    """
    add the config to the root logger.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    # start the queue handler thread
    # queue_handler = logging.getHandlerByName("queue_handler")
    # if queue_handler is not None:
    #     queue_handler.listener.start()
    #     atexit.register(queue_handler.listener.stop)
