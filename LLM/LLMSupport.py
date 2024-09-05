import sys
sys.path.append("LLM")

import logging
logging.getLogger().setLevel(logging.DEBUG)
from datetime import datetime
import Prompts
import json
import logging
import math
import os
import random
import argparse
from itertools import chain
from pathlib import Path
from typing import Tuple, Dict, List
from peft import LoraConfig, PeftModel, get_peft_model
from peft import AutoPeftModelForCausalLM
import pathlib
import datasets
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
from transformers.generation import TextStreamer

import logging
logging.getLogger().setLevel(logging.DEBUG)


################## REMOVE Repo before commit !!!!!!!!!!!!!!!!!
DEBUGGING_MODE = True
cwd = os.getcwd()
os.environ["LLM_PARAMS_PATH_INFERENCE"] = os.path.join(cwd, "LLM", "llm_params_inference.json")
os.environ["LLM_PARAMS_PATH_TRAINING"] = os.path.join(cwd, "LLM", "llm_params_training.json")

assert os.path.exists(os.environ["LLM_PARAMS_PATH_INFERENCE"]), "File not found: " + os.environ["LLM_PARAMS_PATH_INFERENCE"]
assert os.path.exists(os.environ["LLM_PARAMS_PATH_TRAINING"]), "File not found: " + os.environ["LLM_PARAMS_PATH_TRAINING"]


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
)


from BDDTestingLLM_args import parse_args
#import projsecrets

# The correct way would be to use the get_logger function from accelerate.logging,
# but it is not available in this environment
logger = logging.getLogger("GameUnitTestLLM") #get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class BDDTestingLLM:
    args: argparse.Namespace = None
    accelerator: Accelerator = None
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    pipeline: transformers.pipeline = None
    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    train_dataset: datasets.Dataset = None
    eval_dataset: datasets.Dataset = None
    lr_scheduler: torch.optim.lr_scheduler = None
    optimizer: torch.optim.Optimizer = None
    config: AutoConfig = None
    total_batch_size: int = None
    checkpointing_steps: int = None
    terminators: list = None
    torch_dtype = None
    attn_implementation = None

    curr_conversation: List[Dict[str, str]] = []

    def __init__(self, args: argparse.Namespace):
        self.args = args


    def do_training(self):
        #self.prepare_accelerator()
        self.load_model_and_tokenizer()
        self.prepare_data()
        #self.prepare_training()
        self._do_training()

    def load_model_and_tokenizer(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.args.model_name_or_path,
            model_kwargs={"torch_dtype": torch.bfloat16,
                          #"rope_scaling": {"type": "extended", "factor": 8.0}
                          },
            trust_remote_code=self.args.trust_remote_code,
            #_attn_implementation='eager',
            token=Prompts.USER_HF_TOKEN,
            device_map="auto",

        )


        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name,
                                                           token=Prompts.USER_HF_TOKEN)


        # Set the terminators
        self.terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if torch.cuda.get_device_capability()[0] >= 8:
            self.attn_implementation = "flash_attention_2"
            self.torch_dtype = torch.bfloat16
        else:
            self.attn_implementation = "eager"
            self.torch_dtype = torch.float16

        # TODO: add to args
        if self.args.model_name_or_path:

            # Solve quantization
            bnb_config = None
            if self.args.use_4bit_double_quant or self.args.use_4bit_quant:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=False if self.args.use_4bit_double_quant is None
                    else self.args.use_4bit_double_quant
                )
            elif self.args.use_8bit_quant:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.torch_dtype,
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                quantization_config=bnb_config,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
                device_map="auto",
                #low_cpu_mem_usage=self.args.low_cpu_mem_usage,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.args.trust_remote_code,
                attn_implementation=self.attn_implementation
            )
        else:
            logger.info("Training new model from scratch")
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=self.args.trust_remote_code)

        # Need to distinguish between the pad_token and eos_token otherwise the model will think the eos_token is the pad_token
        # so the LLM will not be able to generate the eos_token correctly.
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "right"

        # Create a LoRA config
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, self.peft_config)
        print(f"Model: {self.model.print_trainable_parameters()}")

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_data(self):
        # Get the datasets:  either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found.
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.
        if self.args.dataset_useonline:
            # Downloading and loading a dataset from the hub.
            dataset_name = "your_huggingface_dataset_name"
            dataset_config_name = "your_huggingface_dataset_config_name"
            raw_datasets = load_dataset(dataset_name, dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(dataset_name,
                                                          dataset_config_name,
                                                          split=f"train[:{self.args.validation_split_percentage}%]",
                                                          )
                raw_datasets["train"] = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            dataset_args = {}
            extension = None
            if self.args.train_file is not None:
                data_files["train"] = self.args.train_file
                extension = self.args.train_file.split(".")[-1]
            if self.args.validation_file is not None:
                data_files["validation"] = self.args.validation_file
                extension = self.args.validation_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = not self.args.no_keep_linebreaks
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

            len_dataset = len_dataset = len(raw_datasets['train']) if 'train' in raw_datasets.keys() else len(raw_datasets)

            start_of_training_percent = self.args.validation_split_percentage # Normally 1..val for validation, rest for training, but see below!
            if math.floor(len_dataset * (self.args.validation_split_percentage / 100)) == 0:
                logger.error(f"The validation split percentage {self.args.validation_split_percentage} "
                             f"or dataset size {len_dataset} is too low. "
                             f"Please increase it. For now the script will use the entire dataset "
                             f"for training and validation")
                self.args.validation_split_percentage = 100 #int(math.ceil(len_dataset * (self.args.validation_split_percentage)) / len_dataset)
                start_of_training_percent = 0


            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{start_of_training_percent}%:]",
                    **dataset_args,
                )

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        instruction_column_name = "instruction"
        answer_column_name = "response"
        assert instruction_column_name in column_names, (f"Column 'instruction' not found in dataset. "
                                                         f"Found columns: {column_names}")
        assert answer_column_name in column_names, (f"Column 'response' not found in dataset. "
                                                      f"Found columns: {column_names}")

        # No longer used in the code, but kept for reference
        def tokenize_function(examples):
            prompts = examples[instruction_column_name] if answer_column_name is not None else examples

            answers = examples[answer_column_name] if answer_column_name is not None else None

            prompt_and_answers_tokenized = []
            for prompt, answer in zip(prompts, answers):
                prompt_and_answer = f"{prompt}\n{answer}"

                res = self.tokenizer(prompt_and_answer)
                prompt_and_answers_tokenized.append(res)

            #print(f"TOKENIZING {examples}. Output = {tokenizer(full_example_str)}")

            return prompt_and_answers_tokenized

        def get_full_prompt_from_instruction_and_response(instruction, response):
            return f"### Instruction:{instruction}\n ### Answer:{response}\n"

        def format_entries(examples, add_answer):
            # Transform the dataset to have a single column called 'text' by providing a tuple for each example
            # using ### Instruction ### Response
            prompts = examples[instruction_column_name] if answer_column_name is not None else examples
            answers = examples[answer_column_name] if answer_column_name is not None else None

            for prompt_idx, prompt_str in enumerate(prompts):
                # If the prompt_str is a list of strings (e.g. a list of instructions), join them together
                if type(prompt_str) == list:
                    prompts[prompt_idx] = "\n".join(prompt_str)

            # The new column data to be added
            text_column = []

            for prompt_str, answer in zip(prompts, answers):
                prompt_and_answer = (
                    get_full_prompt_from_instruction_and_response(prompt_str, answer if add_answer else ""))
                text_column.append(prompt_and_answer)

            return {"text": text_column}

        self.train_dataset = raw_datasets["train"]
        self.eval_dataset = raw_datasets["validation"]

        processing_ds = [self.train_dataset, self.eval_dataset]
        fn_kwargs = [{"add_answer": True}, {"add_answer": False}]

        for idx, (raw_dataset, fn_kwarg) in enumerate(zip(processing_ds, fn_kwargs)):
            raw_dataset = raw_dataset.map(
                format_entries,
                batched=True, #False if DEBUGGING_MODE else True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False, #not self.args.overwrite_cache,
                desc="Running a formatting operation on the dataset",
                fn_kwargs=fn_kwarg
            )
            if idx == 0:
                self.train_dataset = raw_dataset
            elif idx == 1:
                self.eval_dataset = raw_dataset

        if self.args.block_size is None:
            block_size = self.tokenizer.model_max_length
            # if block_size > self.config.max_position_embeddings:
            #     logger.error(
            #         f"The tokenizer picked seems to have a very large `model_max_length` ({self.tokenizer.model_max_length}). "
            #         f"Using block_size={min(1024, self.config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            #     )
            #     block_size = min(1024, self.config.max_position_embeddings)
        else:
            if self.args.block_size > self.tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.args.block_size}) is larger than the maximum length for the model "
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.args.block_size, self.tokenizer.model_max_length)


        # len_train_dataset = len(self.train_dataset)
        # # Log a few random samples from the training set:
        # for index in random.sample(range(len_train_dataset), min(3, len_train_dataset)):
        #     logger.info(f"Sample {index} of the training set: {self.train_dataset[index]}.")
        #
        # # DataLoaders creation:
        # self.train_dataloader = DataLoader(
        #     self.train_dataset, shuffle=True, collate_fn=default_data_collator,
        #     batch_size=self.args.per_device_train_batch_size
        # )
        # self.eval_dataloader = DataLoader(
        #     self.eval_dataset, collate_fn=default_data_collator,
        #     batch_size=self.args.per_device_eval_batch_size
        # )

    def _do_training(self):
        import wandb
        from peft import LoraConfig
        from peft import PeftModel

        wandb.login()
        wandb.init(project="game-unittest-llm")

        max_seq_length = min(self.tokenizer.model_max_length, 1024)

        # Get current time as str


        run_name = f"game-unittest-llm_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        training_args = TrainingArguments(
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=self.args.num_train_epochs,
            eval_strategy="epoch",
            #save_strategy="epoch",
            #load_best_model_at_end=True,
            push_to_hub=False,
            logging_steps=1,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            warmup_steps=5,
            weight_decay=0.01,
            report_to=[self.args.report_to],
            overwrite_output_dir=True,
            output_dir="LLM/outputs",
            #save_embedding_layers=True,
            run_name=run_name,
            seed=42,
        )


        class CustomTrainer(SFTTrainer):
            # Initially I was overriding some functions... but I don't need to do that anymore
            pass

        response_template = "### Answer:"

        # Helper code to debug some tokenization issues
        # first_example = self.train_dataset[0]["text"]
        # first_example_tokenized = self.tokenizer.encode(first_example, add_special_tokens=True)
        # logger.info(f"First example: {first_example}. Tokenized {first_example_tokenized}")
        #
        #
        # idx_response_template = first_example.find(response_template)
        # if idx_response_template == -1:
        #     assert False, f"Response template not found in the first example: {first_example}"
        # fist_example_up_to_response_template = first_example[:idx_response_template + len(response_template)]
        # fist_example_up_to_response_template_tokenized = self.tokenizer.encode(fist_example_up_to_response_template, add_special_tokens=True)
        # logger.info(f"First example up to response template: {fist_example_up_to_response_template}. Tokenized {fist_example_up_to_response_template_tokenized}")
        #
        # only_response_template_tokenized = self.tokenizer.encode(response_template, add_special_tokens=False)
        # logger.info(f"Only response template tokenized: {only_response_template_tokenized}")

        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer, response_template=response_template)

        # Initialize our Trainer
        trainer = CustomTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=data_collator,
            args=training_args,
        )

        # Training
        if self.args.do_train:
            trainer.train(
                resume_from_checkpoint=self.args.resume_from_checkpoint)

        # Evaluation
        results = {}
        if self.args.do_eval:
            logger.info("*** Evaluate ***")
            results = trainer.evaluate()

        return results

    def prepare_inference(self, push_to_hub=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.pretrained_peft_adapter_dir is None or self.args.pretrained_peft_adapter_dir == "":
            self.load_model_and_tokenizer()
        else:
            self.model = AutoPeftModelForCausalLM.from_pretrained(self.args.pretrained_peft_adapter_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_peft_adapter_dir)

            self.pipeline = transformers.pipeline(
            "text-generation",
                model=self.model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

        if push_to_hub:
            self.model.push_to_hub("yourhfusername/youradaptername")

        # Init the conversation history
        self.append_system_message("You are a BDD testing expert that can write scenarios "
                                                  "using Gherkin language, Python language and behave library")

    def append_system_message(self, message: str):
        self.curr_conversation.append({"role": "system", "content": message})

    def append_user_message(self, message: str):
        self.curr_conversation.append({"role": "user", "content": message})

    def append_assistant_message(self, message: str):
        self.curr_conversation.append({"role": "assistant", "content": message})

    # Use story_history = False if you don't want the LLM output to be stored in the conversation history
    def do_inference(self, prompt: str, max_generated_tokens, store_history: bool) -> str:
        self.append_user_message(prompt)

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(self.curr_conversation,
                      max_new_tokens=max_generated_tokens,
                      do_sample=False,
                      #temperature=0.1,
                      #top_p=0.9,
                      )

        return_res = outputs[0]["generated_text"][-1]

        if store_history:
            self.append_assistant_message(return_res)
        else:
            self.curr_conversation = self.curr_conversation[:-1] # We remove the last entry added above

        return return_res

