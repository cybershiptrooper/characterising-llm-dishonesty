import logging
import os
import time

DEBUG = True

storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
file = os.path.join(storage_dir, '{}{}.log'.format(
                                "DEBUG_" if DEBUG else "", 
                                time.strftime('%d-%m-%H-%M-%S')
                            ))
open(file, 'w').close()
logging.basicConfig(filename=file, level=logging.INFO, format='')

def log(msg):
    logging.info(msg)
    print(msg)

def log(*args, sep=' ', end='\n'):
    msg = sep.join([str(arg) for arg in args]) + end
    logging.info(msg)
    print(msg, sep=sep, end=end)

if __name__ == '__main__':
    log('test')