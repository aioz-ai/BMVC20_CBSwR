import logging


def get_logger(name, log_file):
    logger = logging.getLogger(name)
    # Setup logging
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG,
                        format="%(message)s",
                        filemode='w')
    return logger
