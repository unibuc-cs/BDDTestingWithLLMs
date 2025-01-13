### Check to see if it works with LLAMA 3.1 :
# https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph

from LLM.LLMSupport import *
from typing import Union, Tuple, List, Dict, Any, Optional
import os
from pprint import pprint
import json
from LLM.utils import fill_regex_with_dictvalues, extract_clauses
from enum import StrEnum

import transformers as tr
#print(tr.__version__)


class GherkinStepType(StrEnum):
    GIVEN = "GIVEN"
    WHEN = "WHEN"
    THEN = "THEN"
    AND = "AND"
    BUT = "BUT"

    def __str__(self):
        return self.value

class BDDGenerationWithLLM:
    def __init__(self):
        # Read params and instantiate the LLM
        self._source_code_clauses = None
        self._source_code_context = None
        args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_INFERENCE"]))
        self.llm = BDDTestingLLM(args)
        self.llm.prepare_inference(push_to_hub=False)
        

    # The BIG TODO :)
    def generate_bdd(self, prompt):
        # Generate a BDD test
        #self.llm.do_inference(prompt=prompt)
        pass

    def get_conv_hist(self) -> List[Dict]:
        return self.llm.full_conv_history
    
    def stack_current_conversation(self):
        self.llm.full_conv_history.stack_current_conversation()

    def pop_current_conversation(self):
        self.llm.full_conv_history.pop_current_conversation()

    def reset_conversation(self):
        self.llm.full_conv_history.reset()

    def get_llm(self) -> tr.AutoModelForCausalLM:
        return self.llm.model

    def get_llm_pipe(self) -> tr.Pipeline:
        return self.llm.pipeline

    # This is the result of the matching.
    # It contains the step found and the parameters matched in plain strings
    # The parameters are in a dictionary with the key being the parameter name and the value the matched value
    # If the step was not found, the step_found is None
    # If the parameters were not matched, the params_matched is None
    # The step_msg and params_msg are the raw outputs from the model
    class MatchingInputResult:
        def __init__(self):
            self.step_found : Union[str, None] = None
            self.step_msg : str = ""

            self.params_matched : Union[Dict[str,Any], None] = None
            self.params_msg : str = ""
            self.filled_step : str = ""
            self.warning_msg : str = ""


        def __repr__(self):
            if self.step_found is None:
                return f"I was unable to identify the step. The raw output is: {self.step_msg}"
            else:
                if self.params_matched is None:
                    return (f"The step found is: {self.step_found}.\n"
                            f"But I was unable to identify the parameters. The raw output is: {self.params_msg}\n"
                            f"WARNINGS: {self.warning_msg}\n" if self.warning_msg.strip() != "" else "")
                else:
                    return (f"The step found is: {self.step_found}.\n"
                            f"The parameters matched are: {self.params_matched}.\n"
                            f"The filled step is: {self.filled_step}.\n"
                            f"WARNINGS: {self.warning_msg}\n" if self.warning_msg.strip() != "" else "")

    # NOTE: this is the internal func, the public is below this
    # Takes an input step in natural language and tries to match against a set of available input steps in a Gherkin file.
    # Two steps are used in the process:
    # Step 1: try to find the closest in terms of matching
    # Step 2: try to match parameters. Report the error if not succeeded
    def _match_input_step_to_set(self,
                                 USER_INPUT_STEP_CONTENT: str,
                                 USER_INPUT_STEP_TYPE: GherkinStepType,
                                 USER_INPUT_AVAILABLE_STEPS: str,
                                 max_generated_tokens: int = 1024) -> MatchingInputResult:

        result = self.MatchingInputResult()

        match_rewrite_prompt_template = """Given the below Gherkin available steps, check if any {user_step_type} step is close to the input step and return it. Do not write code, just provide the step you found.

        Use Json syntax for response. Use the following format if any step can be matched:
        {{
          "found": true,
          "step_found":  the step you found
        }}

        If no available option is OK, then use:
        {{
            "found": false,
        }}

        Do not provide any other information or examples.

        ### Input step: 
        {user_input_step_to_match}

        ### Available steps:
        {user_input_available_steps}"""

        prompt_matching_params_template = """Given the gherkin step in the target, your task is to extract the parameters and assign them values using the input sentence.
        Do not write code, just provide the values in a Json format as in the example below. 

        Example:
        ### Input: The plane has a travel speed of 123 km/h and a length of 500 m
        ### Target: @given(A plane that has a (?P<speed>\d+) km/h, length (?P<size>\d+) m
        Response:
        {{
         "speed" : "123 km/h",
         "size" : "500 m"
        }}
        
        Another example with template variables (e.g., <variable>) that you need to copy in the result as below if used:
        ### Input: The plane has a travel speed of <speed> km/h and a length of <size> m
        ### Target: @given(A plane that has a (?P<speed>\d+) km/h, length (?P<size>\d+) m
        Response:
        {{
         "speed" : <speed>,
         "size" : <size>,
        }}

        Your task:
        ### Input: {user_input_step}
        ### Target: {step_str}

        Response:    your response 

        Do not write anything else.
        """

        match_rewrite_prompt = match_rewrite_prompt_template.format(user_input_step_to_match=USER_INPUT_STEP_CONTENT,
                                                                    user_step_type= str(USER_INPUT_STEP_TYPE), #"USER_INPUT_STEP_TYPE,
                                                                    user_input_available_steps=USER_INPUT_AVAILABLE_STEPS)


        res_match_step = self.llm.do_inference(match_rewrite_prompt,
                                               max_generated_tokens,
                                               store_history=False)["content"]
        res_match_step = res_match_step.replace("\\", "\\\\")  # escape the backslashes
        res_match_step = res_match_step.replace("\'\"", "\"")  # replace the single + double quotes with double quotes
        res_match_step = res_match_step.replace("'\n", "\n")  # replace the single + new line with nothing
        logger.warning(f"Plain result: {res_match_step}")

        # Loading in json
        res_json = None
        try:
            dir = json.loads(res_match_step)
            res_json = dir
        except Exception as e:


            msg = f"Error {e} when parsing the raw output:\n{res_match_step}"
            logger.warning(msg)

            result.step_msg = msg
            return result

        step_found_str = res_json.get("step_found", None)
        result.step_found = step_found_str

        if step_found_str is not None:
            step_found = step_found_str.replace("\\\\", "\\") # unescape the backslashes

            prompt_matching_params = prompt_matching_params_template.format(user_input_step=USER_INPUT_STEP_CONTENT,
                                                                            step_str=step_found_str)

            res_match_params = self.llm.do_inference(prompt_matching_params,
                                                     max_generated_tokens,
                                                     store_history=False)["content"]
            res_match_params = res_match_params.replace("\\", "\\\\")  # escape the backslashes

            # RESPONSE_TAG = "Response:"
            # resp_json_begin_index = res22.find(RESPONSE_TAG) + len(RESPONSE_TAG)
            resp_json_begin_index = res_match_params.find("{")
            resp_json_end_index = res_match_params.rfind("}")

            if resp_json_begin_index != -1 and resp_json_end_index != -1:
                resp_json_str = res_match_params[resp_json_begin_index: resp_json_end_index + 1]

                try:
                    resp_json = json.loads(resp_json_str)


                    # Fill the step parameters found in the step step_found_str with the values found in resp_json
                    msg = "The model found the parameters: {temp_resp}\n".format(temp_resp=resp_json)

                    is_succeed, op_msg, warning_msg = fill_regex_with_dictvalues(step_found_str, resp_json)
                    if is_succeed:
                        msg += f"The filled step is: {op_msg}\n"
                    else:
                        msg += f"Error when filling the step: {op_msg}\n"

                    if warning_msg.strip() != "":
                        msg += f"Warnings:\n {warning_msg}\n"

                    result.params_matched = resp_json
                    result.params_msg = msg
                    result.filled_step = op_msg
                    result.warning_msg = warning_msg

                    logger.warning(resp_json)
                    return result
                except Exception as e:
                    msg = f"Error {e} when parsing for parameters:\n{resp_json_str}"
                    pprint(msg)

                    result.params_msg = msg
                    return result

            msg = "The model didn't a valid match. The raw output is {temp_resp}".format(temp_resp=res_match_params)
            logger.warning(msg)
            result.params_msg = msg
            return result
        else:
            msg = ("Could not find the step matched, i.e., the string step_found. "
                   "The raw output is {step_found_str}").format(step_found_str=step_found_str)
            logger.debug(msg)
            result.params_msg = msg
            return result

        return result

    # Takes an input step in natural language and tries to match against a set of available input steps in a Gherkin file.
    # Two steps are used in the process:
    # Step 1: try to find the closest in terms of matching
    # Step 2: try to match parameters. Report the error if not succeeded
    def match_input_step_to_set(self,
                                USER_INPUT_STEP_CONTENT: str,
                                USER_INPUT_STEP_TYPE : GherkinStepType,
                                USER_INPUT_AVAILABLE_STEPS: Union[str, None],
                                max_generated_tokens: int = 1024) -> MatchingInputResult:

        key = None
        match USER_INPUT_STEP_TYPE:
            case GherkinStepType.GIVEN:
                key = "given"
            case GherkinStepType.THEN:
                key = "then"
            case GherkinStepType.WHEN:
                key = "when"

        assert key is not None, "No valid step type was provided"

        USER_INPUT_AVAILABLE_STEPS = USER_INPUT_AVAILABLE_STEPS \
            if USER_INPUT_AVAILABLE_STEPS else self._source_code_clauses[key]
        assert USER_INPUT_AVAILABLE_STEPS is not None, "No valid steps to match against user input were provided"

        match_rest: BDDGenerationWithLLM.MatchingInputResult = (self._match_input_step_to_set
                                                                 (USER_INPUT_STEP_CONTENT = USER_INPUT_STEP_CONTENT,
                                                                  USER_INPUT_STEP_TYPE=GherkinStepType.GIVEN,
                                                                    USER_INPUT_AVAILABLE_STEPS=USER_INPUT_AVAILABLE_STEPS,
                                                                max_generated_tokens=max_generated_tokens))

        return match_rest

    def set_source_code_context(self, source_code_dirs: List[Path]) -> None:
        self._source_code_context = source_code_dirs
        self._source_code_clauses = {}
        for path in source_code_dirs:
            clauses = extract_clauses(str(path))
            self._source_code_clauses.update(clauses)


if __name__ == '__main__':
    bdd_gen = BDDGenerationWithLLM()
    bdd_gen.set_source_code_context([Path("./car-behave-master")])

    #USER_INPUT_STEP_CONTENT = "A drag of 123, a mass of 12345 kg, and an engine of 124kw the Yoda's vehicle has!"

    USER_INPUT_STEP_CONTENT = "A drag coefficient of <drag>, a mass of <mass> kg, and an engine of <power> kw the Yoda's vehicle has!"

    # TODO: take from file with source code
    #
    # USER_INPUT_AVAILABLE_STEPS = """@given("the car has (?P<engine_power>\d+) kw, weighs (?P<weight>\d+) kg, has a drag coefficient of (?P<drag>[\.\d]+)")
    #
    # @given("a frontal area of (?P<area>.+) m\^2")
    #
    # @when("I accelerate to (?P<speed>\d+) km/h")
    #
    # @then("the time should be within (?P<precision>[\d\.]+)s of (?P<time>[\d\.]+)s")
    #
    # @given("that the car is moving at (?P<speed>\d+) m/s")
    #
    # @when("I brake at (?P<brake_force>\d+)% force")
    #
    # @step("(?P<seconds>\d+) seconds? pass(?:es)?")
    #
    # @then("I should have traveled less than (?P<distance>\d+) meters")
    #
    # @given("that the car's heading is (?P<heading>\d+) deg")
    #
    # @when("I turn (?P<direction>left|right) at a yaw rate of (?P<rate>\d+) deg/sec for (?P<duration>\d+) seconds")
    #
    # @then("the car's heading should be (?P<heading>\d+) deg")"""
    #


    match_res : BDDGenerationWithLLM.MatchingInputResult = bdd_gen.match_input_step_to_set(USER_INPUT_STEP_CONTENT = USER_INPUT_STEP_CONTENT,
                                                                                             USER_INPUT_STEP_TYPE=GherkinStepType.GIVEN,
                                                                                             USER_INPUT_AVAILABLE_STEPS=None)
    logger.warning(f"Final result: {match_res}")
