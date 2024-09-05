from LLM.LLMSupport import *
import pathlib
import os
from BDDTestingLLM_args import parse_args
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_TRAINING"]))
    cg = BDDTestingLLM(args)
    cg.do_training()
