import logging
import os
import time

file = ""
DEBUG = False
storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

def set_logger():
    global file
    file = os.path.join(storage_dir, '{}{}.log'.format(
                                "DEBUG_" if DEBUG else "", 
                                time.strftime('%d-%m-%H-%M-%S')
                            ))
    open(file, 'w').close()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=file, level=logging.INFO, format='')


def set_debug(debug):
    global DEBUG
    DEBUG = debug
    if file != "":
        # delete old file
        os.remove(file)
    set_logger()

def get_debug():
    return DEBUG

def log(msg):
    logging.info(msg)
    print(msg)

def log(*args, sep=' ', end='\n', dont_print=True):
    msg = sep.join([str(arg) for arg in args]) + end
    logging.info(msg)
    if not dont_print:
        print(msg, sep=sep, end=end)

set_logger()