# A file containing the suite of functions that are used during the conversation with the LLM model.
import json
import os.path
from typing import Any, Dict, Tuple, Union, List
from BDDToLLM import BDDGenerationWithLLM
from enum import IntEnum
import LLM.utils as utils
import inspect, types
from LLM.LLMSupport import BDDTestingLLM, AgentTypePrompt, LLMConvHistory
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

                                                  
BASE_SYSTEM_MESSAGE = """You are a BDD testing expert who can write scenarios using the Gherkin language, Python and the behave library."""

BASE_SYSTEM_MESSAGE_WITH_TOOLS_CALL = BASE_SYSTEM_MESSAGE + """If possible, use the tools provided first. 
If no function is suitable for the user query, answer without tools, as if you were the expert. """


# The global instance of the conversation context
globalInstanceBDDAssistant = None #Union[BDDAssistant, None] = None    
    
# This class retains the context of the conversation with the LLM model
class BDDAssistant:
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            # Init the instance
            cls.instance = super(BDDAssistant, cls).__new__(cls)
            
            global globalInstanceBDDAssistant
            assert globalInstanceBDDAssistant is None, "The global instance of the conversation context is already set. Please let it set only here"
            globalInstanceBDDAssistant = cls.instance
            
            # Register all tools use operations
            #---------------------------------
            cls.all_tool_use_operations = {}
                
            # Get the module instance of the globalInstanceBDDAssistant
            currModuleINst = inspect.getmodule(cls.instance)

            # Get all the global function in the module    
            for name, obj in inspect.getmembers(currModuleINst, inspect.isfunction):
                if name.startswith("global_tool_"): # Syntax prefix for all tool use operations
                    cls.all_tool_use_operations[name] = obj
        #---------------------------------
        
            
        return cls.instance
    
        # global globalInstanceBDDAssistant#
        # if globalInstanceBDDAssistant is None:
        #     globalInstanceBDDAssistant = super(BDDAssistant, cls).__new__(cls)
        # return globalInstanceBDDAssistant
     
    def __init__(self, bddGenerationCore: BDDGenerationWithLLM):
        # The current conversation context
        self.conv_state: ConversationState = ConversationState(
            user_scenario="",
            clauses={'given': [], 'when': [], 'then': []},
            current_step_type=None,
            step_impl_folders=[],
            test_case_path=""
        )
            

        self.bddGenerationCore = bddGenerationCore

        self.llm_pipe = self.bddGenerationCore.get_llm_pipe()
        self.llm_support = self.bddGenerationCore.llm
        
        
        #self._hook_execute_tool_use_operation = None
        #self._hook_get_all_tool_use_operations = None
        
        self.init_conv_history()
        
    # Reset the conversation history and the current conversation and set the base system message
    # Can be used to override the base system message
    def init_conv_history(self):        
        # Init the conversation history
        DEFAULT_SYSTEM_MESSAGE = BASE_SYSTEM_MESSAGE_WITH_TOOLS_CALL if self.all_tool_use_operations is not None and len(self.all_tool_use_operations) > 0 else BASE_SYSTEM_MESSAGE
        self.llm_support.full_conv_history.set_base_system_prompt(DEFAULT_SYSTEM_MESSAGE)
        #self.conv_hist : List[Dict] = self.bddGenerationCore.get_conv_hist()
        
    # This function is used to execute a tool use operation given its name and args
    # Returns a tuple of the tool use operation name and the result of the tool use operation
    @staticmethod
    def execute_tool_use_operation(tool_use_op: str, *args) -> Any: # Tuple[str, Any]:
        """
        Execute a tool use operation based on the given tool use operation string.
        
        Args:
            tool_use_op: The tool use operation to execute.
            
        Returns:
            A string containing the result of the tool use operation.
        """ 

        if tool_use_op not in BDDAssistant.all_tool_use_operations:
            raise Exception("The tool use operation is not recognized")
         
        ### Call the tool with the parameters
        if args is None or len(args) == 0 or args[0] is None or len(args[0]) == 0:
            result = BDDAssistant.all_tool_use_operations[tool_use_op]()
        else:
            result = BDDAssistant.all_tool_use_operations[tool_use_op](*args)
        
        # Interpret the result maybe with a functor or a lambda function?!
        # Putting the raw result for now
        # res = BDDTestingLLM.format_tool_response(tool_func_name=tool_use_op, tool_func_result=result)
        # print("### Tool result after execution: ", res)
        return result #tool_use_op, result
        
    
    def test_func_call_correctness(self, func_call: str, expected_tool_call: str):
        assert func_call == expected_tool_call, f"Expected tool call {expected_tool_call} but got {func_call}"
        
        
        
    def do_graph_inference(self, msg: str, store_history: bool, max_generated_tokens: int = 512):
        # This will do inference on the langchain graph model
        pass
        
        
    # Do inference with the LLM model
    # The store_history parameter is used to determine if the current conversation should be stored in the conversation history
    # The agent_mode parameter is used to determine the type of agent that should be used to generate the response
    # If a new history object is passed, it will be used temporarly during the operation then reverted back to the original history
    def do_inference(self, msg: str, 
                     store_history: bool, 
                     max_generated_tokens: int = 512,
                     agent_mode: AgentTypePrompt = AgentTypePrompt.AGENT_TYPE_PROMPT_GENERAL_CONV,
                     overriden_conv_history: LLMConvHistory = None):
        
        # Temporarily override the conversation history and use the new one
        if overriden_conv_history is not None:
            temp_conv_hist = self.llm_support.full_conv_history
            self.llm_support.full_conv_history = overriden_conv_history
        
        # Stack the current conversation            
        #if not store_history:
        #    initial_conv = self.get_conv_hist()
        try:                     
            raw_output, tools_called = self.llm_support.do_inference(msg, 
                                                                     max_generated_tokens=max_generated_tokens,
                                                                    store_history=store_history, 
                                                                    agent_prompt_type=agent_mode,
                                                                    tools_to_use=self.all_tool_use_operations if agent_mode == AgentTypePrompt.AGENT_TYPE_PROMPT_TOOL_CALL else None,                                              
                                                                    tools_executor_func= BDDAssistant.execute_tool_use_operation if agent_mode == AgentTypePrompt.AGENT_TYPE_PROMPT_TOOL_CALL else None)
                                                            
            print(f"### Raw assistent output:\n {raw_output}\nTools called: {tools_called}")
            
            #if not store_history:
            #    end_conv = self.get_conv_hist()
        except Exception as e:
            print(f"### Error in command_test_from_user: {e}")
            raise e         
        finally:
            if overriden_conv_history is not None:
                self.llm_support.full_conv_history = temp_conv_hist
            pass        

    def get_conv_hist(self) -> List[Dict]:
        return self.bddGenerationCore.get_conv_hist()
    
    def stack_current_conversation(self):
        self.bddGenerationCore.stack_current_conversation()
        
    def pop_current_conversation(self):
        self.bddGenerationCore.pop_current_conversation()
        
    def append_user_message(self, message: str):
        self.bddGenerationCore.llm.append_user_message(message)
        
    def append_assistant_message(self, message: str):
        self.bddGenerationCore.llm.append_assistant_message(message)

    ### Automatic messages for the user to know what to do next
    ###--------------------------------------------------------
    def get_current_step_msg(self):
        match self.conv_state.current_step_type:
            case ConvStepType.SCENARIO_DEF_PART1_SCENARIO_DEF:
                return "Please provide a scenario definition in natural language."
            case ConvStepType.SCENARIO_DEF_PART2_SCENARIO_OUTPUTPATH:
                return "Provide the path where the test case will be saved."
            case ConvStepType.SCENARIO_DEF_PART3_CODE_PATH:
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
        match self.conv_state.current_step_type:
            case ConvStepType.SCENARIO_DEF_PART1_SCENARIO_DEF:
                self.current_step_type = ConvStepType.GIVEN
            case ConvStepType.SCENARIO_DEF_PART2_SCENARIO_OUTPUTPATH:
                self.current_step_type = ConvStepType.SCENARIO_DEF_PART3_CODE_PATH
            case ConvStepType.SCENARIO_DEF_PART3_CODE_PATH:
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
            
        def __repr__(self) -> str:
            return f"ToolUseOpResult(success={self.success}, message={self.message})"

    # This function is used to extract the path from the user message
    def _tool_find_path_from_user(self) -> ToolUseOpResult:
        """
        Given the last user message, extract a path from the user message that corresponds to the path of a file where the test will be saved.
        Args:
            self (BDDAssistant): The instance of the class.
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successful and message is the list of paths.
            
        """

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
        
        assert res is not None, "The response from the LLM model was None"
        assert res['role'] == 'assistant', "The response from the LLM model was not from the assistant"
        res = res['content']
        try:
            json_res = json.loads(res)
            if "path" not in json_res:
                return BDDAssistant.ToolUseOpResult(False, "The path was not found in the response")
            
            msg_res = json_res["path"]

            if utils.is_pathname_valid(msg_res):
                return BDDAssistant.ToolUseOpResult(True, msg_res)
            else:
                return BDDAssistant.ToolUseOpResult(False, f"The path provided, {msg_res} is not valid")

        except json.JSONDecodeError:
            return BDDAssistant.ToolUseOpResult(False, "The response was not in a valid JSON format")

    # This function is used to extract a list of paths from the user message
    def _tool_find_path_lists_from_user(self) -> ToolUseOpResult:
        """
        Given the last user message, extract a list of paths from the user message.
        Args:
            self (BDDAssistant): The instance of the class.
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successfulm and message is the list of paths.
        """

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
        assert res is not None, "The response from the LLM model was None"
        assert res['role'] == 'assistant', "The response from the LLM model was not from the assistant"
        res = res['content']
        
        try:
            msg = json.loads(res)
            if "paths" not in msg:
                return BDDAssistant.ToolUseOpResult(False, "No paths were found in the response")

            all_paths = msg["paths"]
            if not isinstance(all_paths, list):
                return BDDAssistant.ToolUseOpResult(False, f"Internal error: the paths should be in a list format. They were captured from user input as {all_paths}")

            # Check if the paths are valid
            for path in all_paths:
                if not isinstance(path, str) or os.path.exists(path):
                    return BDDAssistant.ToolUseOpResult(False, f"The paths should be strings "
                                                                      f"and should exist on your disk. \n"
                                                                      f"For example {path} does not exist.")

            return BDDAssistant.ToolUseOpResult(True, str(all_paths))

        except json.JSONDecodeError:
            return BDDAssistant.ToolUseOpResult(False, "The response was not in a valid JSON format")

    # This function is used to build the current test case
    def _tool_get_current_test(self) -> ToolUseOpResult:
        """
        Get the current test as a string in the format of a Behave test case.
        
        Args:
            self (BDDAssistant): The instance of the class.
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successful and message is the test case string.
        """
        
        full_test_text: str = "Scenario Outline: " + self.conv_state.get_user_scenario() + "\n"

        # Write the Given clauses
        clauses_order = ['given', 'when', 'then']
        for cl_type in clauses_order:
            for cl_idx, clause_str in enumerate(self.conv_state.get_clauses(cl_type)):
                if cl_idx == 0:
                    full_test_text += cl_type.capitalize() + " "
                else:
                    full_test_text += "And "

                full_test_text += clause_str + "\n"

        return BDDAssistant.ToolUseOpResult(True, full_test_text)

    def _tool_save_current_test(self) -> ToolUseOpResult:
        """
        Saves the current test to a file that was specified by the user in the conversation.
        
        Args:
            self (BDDAssistant): The instance of the class.
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successful and message is the test case string.
        """
        # Get the current test
        res = self._tool_get_current_test()
        if not res.success:
            return BDDAssistant.ToolUseOpResult(False, "An error occurred while getting the current test. original error: " + res.message)

        # Save the test to the file
        res_test_retr = self._tool_get_current_test()
        test_text = res_test_retr.message
        try:
            with open(self.conv_state.test_case_path, "w") as f:
                f.write(test_text)
        except Exception as e:
            return BDDAssistant.ToolUseOpResult(False, f"An error occurred while saving the test case: {str(e)}")

        return BDDAssistant.ToolUseOpResult(True, "The test case was saved successfully to the specified path, " + self.conv_state.test_case_path)

    def _tool_clear_current_test(self) -> ToolUseOpResult:
        """
        Clears, discards the current test in progress and removes the file from disk if it was saved previously.
        
        Args:
            self (BDDAssistant): The instance of the class.    
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successfulm and message is the test case string.
        """
        self.conv_state.clear_clauses()
        if os.path.exists(self.conv_state.test_case_path):
            os.remove(self.conv_state.test_case_path)
        return BDDAssistant.ToolUseOpResult(True, "The current test case was cleared successfully and removed from disk path if saved previously")

    def _tool_run_test(self) -> ToolUseOpResult:
        """
        Runs or execute the test using pytest or behave.
        Args:
            self (BDDAssistant): The instance of the class.     
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successful and message is the test case string.
        """
        # Get the current test and save it
        res = self.tool_save_current_test()
        if res.success is False:
            return BDDAssistant.ToolUseOpResult(False, "Could not save the current test to the path given. Full error: " + res.message)

        # Run the test

        try:
            import subprocess

            # Run the test
            res = subprocess.run(["behave", "-i", "basic_driving.feature"], capture_output=True, text=True)
            if res.returncode == 0:
                return BDDAssistant.ToolUseOpResult(True, res.stdout)
            else:
                return BDDAssistant.ToolUseOpResult(False, "Error occurred while running the behave tool:\n" + res.stdout + "\n" + res.stderr)

        except Exception as e:
            return BDDAssistant.ToolUseOpResult(False, f"An error occurred while running the behave tool: {str(e)}")
 
    def _tool_add_new_empty_step(self, clause: str = None, filepath: str = None) -> ToolUseOpResult:
        """
        Add a new empty step to the step implementation file. The clause is the step to be added and the filepath is the path to the step implementation file. The Filepath should be a path to a Python file.
        Args:
            self (BDDAssistant): The instance of the class.
            clause: The clause to be added as a new step.
            filepath: The path to the step implementation file
        Returns:
            A tuple of (success, message) where success is a boolean indicating if the operation was successful and message is the test case string.
        """
        
        if clause is None or filepath is None:
            return BDDAssistant.ToolUseOpResult(False, f"The clause given {clause} or filepath {filepath} are invalid")

        clause = clause.strip()
        if not any([clause.find('@given') == 0, clause.find('@when') == 0, clause.find('@then') == 0]):
            return BDDAssistant.ToolUseOpResult(False, f"The clause should start with any of the @type")

        if not os.path.exists(filepath):
            return BDDAssistant.ToolUseOpResult(False, f"The filepath given does not exist {filepath}")



        # Open the file in append mode and add the new proposed step as empty
        try:
            with open(filepath, 'a') as file:
                file.write(f"\n{clause}\ndef step_impl(context):\n\tassert False, 'Not Implemented'\n\tpass\n")
            return BDDAssistant.ToolUseOpResult(True, f"Step '{clause}' added successfully to {filepath}")
        except Exception as e:
            return BDDAssistant.ToolUseOpResult(False, f"An error occurred while adding the step: {str(e)}")
        
    # def get_hook_execute_tool_use_operation(self):
    #     if self._hook_execute_tool_use_operation is None:
    #         raise Exception("The hook for the execute tool use operation is not set")
    #     return self._hook_execute_tool_use_operation
    
    # def get_hook_get_all_tool_use_operations(self):
    #     if self._hook_get_all_tool_use_operations is None:
    #         raise Exception("The hook for the get all tool use operations is not set")
    #     return self._hook_get_all_tool_use_operations


# def hook_set_global_conversation_context(conversation_context: BDDAssistant):
#     global globalInstanceBDDAssistant
#     globalInstanceBDDAssistant = conversation_context
    
# def hook_get_global_conversation_context() -> Union[BDDAssistant, None]:
#     return globalInstanceBDDAssistant

def _hook_check_global_instance():
    assert globalInstanceBDDAssistant is not None, "The global instance of the conversation context is not set. Please set it using the hook_set_global_conversation_context function"
    assert isinstance(globalInstanceBDDAssistant, BDDAssistant), "The global instance of the conversation context is not an instance of the BDDAssistant class"

# The static set of functions that are used to interact with the BDDAssistant model
# This function is used to build the current test case
def global_tool_get_current_test() -> str:
    """
    Get and/or shows the current test as a string in the format of a Behave test case.
    Does not save the test
    
    Args:
     
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_get_current_test()

def global_tool_save_current_test() -> str:
    """
    Saves the current test to a file that was specified by the user in the conversation.
    
    Args:
    
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_save_current_test()

def global_tool_clear_current_test() -> str:
    """
    Clears, discards the current test in progress and removes the file from disk if it was saved previously.
    
    Args:
    
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_clear_current_test()

def global_tool_run_test() -> str:
    """
    Runs,  execute or give a try to see the output of the current test using pytest or behave.
    
    Args:
    
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_run_test()

def global_tool_add_new_empty_step(clause: str, filepath: str) -> str:
    """
    Add a new empty step to the step implementation file. The clause is the step to be added and the filepath is the path to the step implementation file. The Filepath should be a path to a Python file.
    
    Args:
        clause: The clause to be added as a new step.
        filepath: The path to the step implementation file
        
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_add_new_empty_step(clause, filepath)

def global_tool_find_path_from_user() -> str:
    """
    Given the last user message, extract a path from the user message that corresponds to the path of a file where the test will be saved.
    
    Args:
    
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_find_path_from_user()

def global_tool_find_path_lists_from_user() -> str:
    """
    Given the last user message, extract a list of paths from the user message.
    
    Args:
    
    Returns:
        A string containing the test case.
    """
    _hook_check_global_instance()
    return globalInstanceBDDAssistant._tool_find_path_lists_from_user()





# _global_cached_tool_use_operations = None 

# def hook_get_all_tool_use_operations() -> Dict[str, types.FunctionType]:
#     """
#     Get all the tool use operations that are available in the BDDTestingLLM_functions.py file.
    
#     Args: 
#     """
#     global _global_cached_tool_use_operations
#     all_tool_use_operations = {}
     
#     # Get the module instance of the globalInstanceBDDAssistant
#     currModuleINst = inspect.getmodule(globalInstanceBDDAssistant)

#     # Get all the global function in the module    
#     for name, obj in inspect.getmembers(currModuleINst, inspect.isfunction):
#         if name.startswith("global_tool_"):
#             all_tool_use_operations[name] = obj
    
    
#     _global_cached_tool_use_operations = all_tool_use_operations        
#     return all_tool_use_operations


# # This function accept the tool use operation string and execute the corresponding tool use operation
# # plus the arguments that are passed to the function which could be any number of arguments
# def hook_execute_tool_use_operation(tool_use_op: str, *args) -> str:
#     """
#     Execute a tool use operation based on the given tool use operation string.
    
#     Args:
#         tool_use_op: The tool use operation to execute.
        
#     Returns:
#         A string containing the result of the tool use operation.
#     """
#     _hook_check_global_instance()
    
#     assert _global_cached_tool_use_operations is not None, "The global cached tool use operations are not set. Please set them using the hook_get_all_tool_use_operations function"
    
#     if tool_use_op not in _global_cached_tool_use_operations:
#         raise Exception("The tool use operation is not recognized")
    
#     return _global_cached_tool_use_operations[tool_use_op](*args)

