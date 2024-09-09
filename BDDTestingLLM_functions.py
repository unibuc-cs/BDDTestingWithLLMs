# A file containing the suite of functions that are used during the conversation with the LLM model.
import json
import os.path
from typing import Any, Dict, Tuple, Union, List
from BDDToLLM import BDDGenerationWithLLM
from enum import Enum
import LLM.utils as utils


# This enum class defines the different types of steps in a BDD scenario

class ConvStepType(Enum):
    SCENARIO_DEF_PART1 = 0
    SCENARIO_DEF_PART2 = 1
    SCENARIO_DEF_PART3 = 2
    GIVEN = 3
    WHEN = 4
    THEN = 5
    FINE_TUNING = 6


# This class retains the context of the conversation with the LLM model
class ConversationContext:
    def __init__(self, bddGenerationCore: BDDGenerationWithLLM):
        # The current conversation context

        self.user_scenario: str = ""

        # This is a dictionary for the clauses that are being recorded in the conversation
        self.clauses = {'given': [], 'when': [], 'then': []}

        # The current step type
        self.current_step_type: Union[ConvStepType, None] = None

        self.step_impl_folders: List[str] = []

        self.test_case_path: str = ""

        self.bddGenerationCore = bddGenerationCore

        self.conv_hist : List[Dict] = self.bddGenerationCore.get_conv_hist()
        self.llm_pipe = self.bddGenerationCore.get_llm_pipe()
        self.llm_support = self.bddGenerationCore.llm


    def clear_clauses(self):
        self.clauses = {'given': [], 'when': [], 'then': []}

    def set_user_scenario(self, user_scenario):
        self.user_scenario = user_scenario

    def add_clause(self, clause_type, clause):
        self.clauses[clause_type].append(clause)

    def get_clauses(self, clause_type: Any[str, None] = None):
        return self.clauses

    def get_user_scenario(self):
        return self.user_scenario

    ### Automatic messages for the user to know what to do next
    ###--------------------------------------------------------
    def set_current_step_type(self, step_type):
        self.current_step_type = step_type

    def get_current_step_msg(self):
        match self.current_step_type:
            case ConvStepType.SCENARIO_DEF_PART1:
                return "Please provide a scenario definition in natural language."
            case ConvStepType.SCENARIO_DEF_PART2:
                return "Provide the path where the test case will be saved."
            case ConvStepType.SCENARIO_DEF_PART3:
                return ("Provide the folder paths where the step implementations are located. "
                        "If there are multiple folders, separate them with a new line.")
            case ConvStepType.GIVEN:
                return "Please provide a Given clause"
            case ConvStepType.WHEN:
                return "Please provide a When clause"
            case ConvStepType.THEN:
                return "Please provide a Then clause"
            case ConvStepType.FINE_TUNING:
                return "Let's fine-tune the test case"
            case _:
                return "Please provide a clause"

    def move_next_step(self):
        match self.current_step_type:
            case ConvStepType.SCENARIO_DEF_PART1:
                self.current_step_type = ConvStepType.GIVEN
            case ConvStepType.SCENARIO_DEF_PART2:
                self.current_step_type = ConvStepType.SCENARIO_DEF_PART3
            case ConvStepType.SCENARIO_DEF_PART3:
                self.current_step_type = ConvStepType.GIVEN
            case ConvStepType.GIVEN:
                self.current_step_type = ConvStepType.WHEN
            case ConvStepType.WHEN:
                self.current_step_type = ConvStepType.THEN
            case ConvStepType.THEN:
                self.current_step_type = ConvStepType.FINE_TUNING
            case ConvStepType.FINE_TUNING:
                assert False, "The conversation has ended"
            case _:
                assert False, "The conversation has ended"
    ###--------------------------------------------------------

    class ToolUseOpResult:
        def __init__(self, success: bool, message: str):
            self.success = success
            self.message = message

    # This function is used to extract the path from the user message
    def tool_find_path_from_user(self) -> ToolUseOpResult:
        # Temporarily add the prompt request to extract path
        # self.conv_hist.append({"role": "user", "content": """What is the path to the test in the last user message?
        # Respond using a JSON format as follows:
        # {
        #     "path" : your identified path
        # }
        #
        # If you do not find any folder path from the user message, provide an empty string, as below, do not invent one.
        # {
        #     "path : ""
        #
        # }
        #
        # """})

        msg_content = """What is the path to the test in the last user message?
        Respond using a JSON format as follows:
        {
            "path" : your identified path
        }
        
        If you do not find any folder path from the user message, provide an empty string, as below, do not invent one.
        {
            "path : ""        
        }
        """

        res = self.llm_support.do_inference(msg_content, 512, False)
        try:
            msg = json.loads(res)
            if "path" not in msg:
                return ConversationContext.ToolUseOpResult(False, "The path was not found in the response")

            if utils.is_path_valid(msg["path"]):
                return ConversationContext.ToolUseOpResult(True, msg["path"])
            else:
                return ConversationContext.ToolUseOpResult(False, f"The path provided, {msg['path']} is not valid")

        except json.JSONDecodeError:
            return ConversationContext.ToolUseOpResult(False, "The response was not in a valid JSON format")

    # This function is used to extract a list of paths from the user message
    def tool_find_path_lists_from_user(self) -> ToolUseOpResult:
        # Temporarily add the prompt request to extract path
        # self.conv_hist.append({"role": "user", "content": """What is the path to the test in the last user message?
        # Respond using a JSON format as follows:
        # {
        #     "path" : your identified path
        # }
        #
        # If you do not find any folder path from the user message, provide an empty string, as below, do not invent one.
        # {
        #     "path : ""
        #
        # }
        #
        # """})

        msg_content = """Group the files given by the last user message in a JSON file as bellow. 
        Do not write any code or variables, just extract the paths and fill the JSON below.

        {
            "paths" : ["path1", "path2", "path3", ...]
        }

        If you do not find any folder path from the user message, provide a list, as below, do not invent one.
        {
            "paths" : []
        }
        """

        res = self.llm_support.do_inference(msg_content, 512, False)
        try:
            msg = json.loads(res)
            if "paths" not in msg:
                return ConversationContext.ToolUseOpResult(False, "No paths were found in the response")

            all_paths = msg["paths"]
            if not isinstance(all_paths, list):
                return ConversationContext.ToolUseOpResult(False, f"Internal error: the paths should be in a list format. They were captured from user input as {all_paths}")

            # Check if the paths are valid
            for path in all_paths:
                if not isinstance(path, str) or os.path.exists(path):
                    return ConversationContext.ToolUseOpResult(False, f"The paths should be strings "
                                                                      f"and should exist on your disk. "
                                                                      f"For example {path} does not exist.")

            return ConversationContext.ToolUseOpResult(True, msg["path"])

        except json.JSONDecodeError:
            return ConversationContext.ToolUseOpResult(False, "The response was not in a valid JSON format")

