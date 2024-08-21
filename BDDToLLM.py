from LLM.LLMSupport import *
from typing import Union, Tuple, List, Dict, Any, Optional
import os
from pprint import pprint


import transformers as tr
print(tr.__version__)

class BDDGenerationWithLLM:
    def __init__(self):
        # Read params and instantiate the LLM
        args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_INFERENCE"]))
        self.llm = BDDTestingLLM(args)
        self.llm.prepare_inference(push_to_hub=False)

    # The BIG TODO :)
    def generate_bdd(self, prompt):
        # Generate a BDD test
        #self.llm.do_inference(prompt=prompt)
        pass

    # NOTE: this is the internal func, the public is below this
    # Takes an input step in natural language and tries to match against a set of available input steps in a Gherkin file.
    # Two steps are used in the process:
    # Step 1: try to find the closest in terms of matching
    # Step 2: try to match parameters. Report the error if not succeeded
    def _match_input_step_to_set(self,
                                 USER_INPUT_STEP: str,
                                 USER_INPUT_AVAILABLE_STEPS: str,
                                 max_generated_tokens: int = 1024) -> Tuple[bool, str]:

        match_rewrite_prompt_template = """Given the below Gherkin available steps, check if any can match the input step. 

        Use Json syntax for response. Use the following format if any step can be matched:
        {{
          "found": true,
          "step_found":  "the step you found closest"
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

        prompt_matching_params_template = """Can you match the parameters in the input text step with the target step ?

        Example:
        ### Input: The plane has a travel speed of 123 km/h and a lenght of 500 m
        ### Target: @given(A plane that has a (?P<speed>\d+) km/h, length (?P<size>\d+) m
        Response:
        {{
         "speed" : "123 km/h",
         "size" : "500 m"
        }}

        Your task:
        ### Input: {user_input_step}
        ### Target: {step_str}

        Response:    your response 

        Do not write anything else.
        """

        match_rewrite_prompt = match_rewrite_prompt_template.format(user_input_step_to_match=USER_INPUT_STEP,
                                                                    user_input_available_steps=USER_INPUT_AVAILABLE_STEPS)

        import json
        res_match_step = self.llm.do_inference(match_rewrite_prompt, max_generated_tokens)["content"]
        res_match_step = res_match_step.replace("\\", "\\\\")  # escape the backslashes
        pprint(f"Plain result: {res_match_step}")

        # Loading in json
        res_json = None
        try:
            dir = json.loads(res_match_step)
            res_json = dir
        except Exception as e:
            msg = f"Error {e} when parsing the raw output:\n{res_match_step}"
            pprint(msg)
            return (False, msg)

        step_found_str = res_json.get("step_found", None)

        if step_found_str is not None:
            prompt_matching_params = prompt_matching_params_template.format(user_input_step=USER_INPUT_STEP,
                                                                            step_str=step_found_str)
            res_match_params = self.llm.do_inference(prompt_matching_params, max_generated_tokens)["content"]
            res_match_params = res_match_params.replace("\\", "\\\\")  # escape the backslashes

            # RESPONSE_TAG = "Response:"
            # resp_json_begin_index = res22.find(RESPONSE_TAG) + len(RESPONSE_TAG)
            resp_json_begin_index = res_match_params.find("{")
            resp_json_end_index = res_match_params.rfind("}")

            if resp_json_begin_index != -1 and resp_json_end_index != -1:
                resp_json_str = res_match_params[resp_json_begin_index: resp_json_end_index + 1]

                try:
                    resp_json = json.loads(resp_json_str)
                    pprint(resp_json)

                    return (True, resp_json)

                except Exception as e:
                    msg = f"Error {e} when parsing for parameters:\n{resp_json_str}"
                    pprint(msg)

                    return (False, msg)

            msg = "The model didn't a valid match. The raw output is {temp_resp}".format(temp_resp=res_match_params)
            pprint(msg)
            return (False, msg)
        else:
            msg = ("Could not find the step matched, i.e., the string step_found. "
                   "The raw output is {step_found_str}").format(step_found_str=step_found_str)
            pprint(msg)
            return (False, msg)

        assert False, "Should not reach here"
        return (False, None)

    def match_input_step_to_set(self,
                                USER_INPUT_STEP: str,
                                max_generated_tokens: int = 1024) -> Tuple[bool, str]:

        # TODO: take from file with source code
        USER_INPUT_AVAILABLE_STEPS = """@given("the car has (?P<engine_power>\d+) kw, weighs (?P<weight>\d+) kg, has a drag coefficient of (?P<drag>[\.\d]+)")

        @given("a frontal area of (?P<area>.+) m\^2")

        @when("I accelerate to (?P<speed>\d+) km/h")

        @then("the time should be within (?P<precision>[\d\.]+)s of (?P<time>[\d\.]+)s")

        @given("that the car is moving at (?P<speed>\d+) m/s")

        @when("I brake at (?P<brake_force>\d+)% force")

        @step("(?P<seconds>\d+) seconds? pass(?:es)?")

        @then("I should have traveled less than (?P<distance>\d+) meters")

        @given("that the car's heading is (?P<heading>\d+) deg")

        @when("I turn (?P<direction>left|right) at a yaw rate of (?P<rate>\d+) deg/sec for (?P<duration>\d+) seconds")

        @then("the car's heading should be (?P<heading>\d+) deg")"""

        is_matched, resp = self._match_input_step_to_set(USER_INPUT_STEP,
                                                         USER_INPUT_AVAILABLE_STEPS,
                                                         max_generated_tokens=1024)

        return is_matched, resp


if __name__ == '__main__':
    bdd_gen = BDDGenerationWithLLM()
    is_matched, resp = bdd_gen.match_input_step_to_set("A drag of 123, a mass of 12345 kg, and an engine of 124kw the Yoda's vehicle has!")
    pprint(is_matched)
    pprint(resp)
    pass