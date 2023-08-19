# package for aggregator
from config_parser import *
import os,logging,time,datetime    


# %%

script_args.time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
logDir = os.path.join(script_args.log_path, "log", script_args.job_name,script_args.time_stamp) 
logFile = os.path.join(logDir, 'log')


def init_logging():
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='(%m-%d) %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logFile, mode='a'),
            logging.StreamHandler()
        ])
    logging.getLogger("transformers.modeling_utils").setLevel(
        logging.ERROR)  # Reduce logging        


def initiate_aggregator_setting():
    init_logging()
    # logging.info("FL Testing in round: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
    #                 .format(round_num, global_virtual_clock, testing_history['perf'][round_num]['top_1'],
    #                         testing_history['perf'][round_num]['top_5'], testing_history['perf'][round_num]['loss'],
    #                         testing_history['perf'][round_num]['test_len']))


initiate_aggregator_setting()
