
"""
    Global Variable
"""
ENABLE_PROFILER = False # the trigger of Timer 

def toggle_profiler():
    global ENABLE_PROFILER
    ENABLE_PROFILER = not ENABLE_PROFILER

"""
    Logger configuration
"""
import logging

logger = logging.getLogger('spiketag')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
