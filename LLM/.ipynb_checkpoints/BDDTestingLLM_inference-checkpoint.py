from LLM.LLMSupport import *
import pathlib
import os
from BDDTestingLLM_args import parse_args
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_INFERENCE"]))
    cg = BDDTestingLLM(args)
    cg.prepare_inference(push_to_hub=False)

    # Do a simple test first

    print("$$ A first simple test: ")
    #cg.do_inference(prompt="Generate a simple unit test for Unity")

    # Then load one from the dataset -warning, this loads the entire dataset into memory but in this case it's small
    cg.prepare_data()
    first_item = cg.eval_dataset[0]
    cg.do_inference(prompt=first_item["text"])
