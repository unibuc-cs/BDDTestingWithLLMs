# This class contains the data model for the conversation state

from enum import IntEnum
from typing import Dict, List, Union, Optional, Iterable
from pydantic import BaseModel
from BDDToLLM import GherkinStepType
import logging
from enum import StrEnum
import os

class AssistantFixedMessage(StrEnum):
    # This is the message that the assistant will use to ask the user for the scenario description
    ask_for_scenario_description = "Please provide a description of the scenario you would like to test."
    
    # This is the message that the assistant will use to ask the user for the scenario output path
    ask_for_scenario_output_path = "Please provide the path where the scenario output will be saved."
    
    # This is the message that the assistant will use to ask the user for the given step
    ask_for_given_step = "Please provide the Given step."
    
    # This is the message that the assistant will use to ask the user for the when step
    ask_for_when_step = "Please provide the When step."
    
    # This is the message that the assistant will use to ask the user for the then step
    ask_for_then_step = "Please provide the Then step."

    # This is the message that the assistant will use to ask the user for the fine tuning step
    ask_for_fine_tuning_step = "Please provide any final edit requests, save and exit, or discard the test."

    # This is the message that the assistant will use to ask the user for the path where the step implementation is located
    ask_for_step_impl_folders = "Please provide the paths to the folder where the step implementations are located."
    # This is the message that the assistant will use to ask the user for the path where the step implementation is located
    
    # This is the message that the assistant will use to ask the user for the path where the step implementation is located
    ask_for_user_confirmation_general = "Please confirm if the request above is correct."
    
    ask_for_user_confirmation_scenario_edit = "Please confirm if you want to modify the scenarion's {what_to_edit}"
    
    ask_for_user_confirmation_bdd_edit = "Please confirm if you want to modify the {what_to_edit} step."
    
    ask_for_user_confirmation_management = "Please confirm if you want to {operation} the scenario {path_of_loading}"

    # This is the message that the assistant will use to ask the user for the path of the code to be tested
    # ask_for_code_path = "Please provide the path of the code to be tested."


# This enum class defines the different types of steps in a BDD scenario
class ConvStepType(IntEnum):
    SCENARIO_DEF_PART1_SCENARIO_DEF = 0
    SCENARIO_DEF_PART2_SCENARIO_OUTPUTPATH = 1
    SCENARIO_DEF_PART3_CODE_PATH = 2
    GIVEN = 3
    WHEN = 4
    THEN = 5
    FINE_TUNING = 6
    ASK_FOR_USER_CONFIRMATION_GENERAL = 7
    ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT = 8, # This is which property of the scenario definition the user wants to edit
    ASK_FOR_USER_CONFIRMATION_BDD_EDIT = 9,  # This is which BDD step the user wants to edit
    ASK_FOR_USER_CONFIRMATION_MANAGEMENT_OP = 10, # This is when the user wants to save, load, or discard the scenario
    ASSISTANT_DOES_NOT_UNDERSTAND = 11,  # This is when the assistant does not understand the user input


# A dictionary that maps from ConvStepType to the assistant message that will be used to prompt the user for the step
assistant_message_map = {
    ConvStepType.SCENARIO_DEF_PART1_SCENARIO_DEF: AssistantFixedMessage.ask_for_scenario_description,
    ConvStepType.SCENARIO_DEF_PART2_SCENARIO_OUTPUTPATH: AssistantFixedMessage.ask_for_scenario_output_path,
    ConvStepType.SCENARIO_DEF_PART3_CODE_PATH: AssistantFixedMessage.ask_for_step_impl_folders,
    ConvStepType.GIVEN: AssistantFixedMessage.ask_for_given_step,
    ConvStepType.WHEN: AssistantFixedMessage.ask_for_when_step,
    ConvStepType.THEN: AssistantFixedMessage.ask_for_then_step,
    ConvStepType.FINE_TUNING: AssistantFixedMessage.ask_for_fine_tuning_step,
    ConvStepType.ASK_FOR_USER_CONFIRMATION_GENERAL: AssistantFixedMessage.ask_for_user_confirmation_general,
    ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT: AssistantFixedMessage.ask_for_user_confirmation_scenario_edit,
    ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT: AssistantFixedMessage.ask_for_user_confirmation_bdd_edit,
    ConvStepType.ASK_FOR_USER_CONFIRMATION_MANAGEMENT_OP: AssistantFixedMessage.ask_for_user_confirmation_management,
}


def any_word_in_sentence(words: Iterable[str], sentence: str, force_lowercase: bool = False) -> bool:
    sentence = sentence.lower() if force_lowercase else sentence
    return any(word in sentence for word in words)

# The intention or decision making output process of the LLM depending on the state of the conversation, internal state, and the user input
class LLMAnalysisIntention(StrEnum):
    # When LLM does not understand the user input
    LLM_INTENTION_UNKNOWN = "llm_intention_unknown"
    
    # When the LLM recognizes an edit situation
    LLM_INTENTION_EDIT_SCENARIO = "llm_intention_edit_scenario"
    LLM_INTENTION_EDIT_BDD_STEP = "llm_intention_edit_step"

    # When the LLM recognizes a management - save, load, or discard situation
    LLM_INTENTION_SAVE_SCENARIO = "llm_intention_save_scenario"
    LLM_INTENTION_DISCARD_SCENARIO = "llm_intention_discard_scenario"
    LLM_INTENTION_LOAD_SCENARIO = "llm_intention_load_scenario"
    
    # According to the flowchart, we have the following intentions
    LLM_INTENTION_ADD_BDD_STEP = "llm_intention_add_step"
    LLM_INTENTION_SCENARION_STEP = "llm_intention_scenario_step"    


class LLMIntentionAnalysisResult(BaseModel):
    # The intention of the user input
    intention: LLMAnalysisIntention = LLMAnalysisIntention.LLM_INTENTION_UNKNOWN
    
    # The bdd step type that the user is trying to edit or add according to the flowchart
    BDD_STEP_TYPE : Optional[None | GherkinStepType] = None
    
    # The scenario property that the user is trying to edit according to the flowchart
    SCENARIO_EDIT_TYPE: Optional[None | ConvStepType] = None     
    
    user_input_raw: Optional[None | str] = None # The raw user input that was received
    
    temp_path: Optional[None | str] = None # The path where the test case will be saved / loaded or something similar, more as a placeholder for various operations
    
    def format_message(self, current_step_type: ConvStepType) -> str:
        res = None 
        if current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT:
            if self.SCENARIO_EDIT_TYPE is not None:
                res = AssistantFixedMessage.ask_for_user_confirmation_scenario_edit.format(what_to_edit=self.SCENARIO_EDIT_TYPE)
            else:
                res = AssistantFixedMessage..format(what_to_edit="scenario")   ####################
                
        elif current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT:
            customized_msg = AssistantFixedMessage.ask_for_user_confirmation_bdd_edit.format(what_to_edit=self.BDD_STEP_TYPE)
        elif current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_MANAGEMENT_OP:
            customized_msg = AssistantFixedMessage.ask_for_user_confirmation_management.format(operation=self.intention, 
                                                                                                   path_of_loading=self.temp_path)
            
        elif current_step_type == ConvStepType.ASSISTANT_DOES_NOT_UNDERSTAND:
            if self.intention == LLMAnalysisIntention.LLM_INTENTION_UNKNOWN:
                res = "I'm sorry, I did not understand your input." ####################
                
        elif current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT
                
                
            return customized_msg  
        
        return message.format(what_to_edit=self.SCENARIO_EDIT_TYPE, bdd_step=self.BDD_STEP_TYPE, operation=self.intention, path_of_loading=self.temp_path)
    
     
# This class retains the state of the conversation with the LLM model
class ConversationState(BaseModel): # TypedDict):
    # The overall user scenario description in natural language
    user_scenario: str
    
    # This is a dictionary for the clauses that are being recorded in the conversation
    clauses: Dict[str, List[str]]
    # The current step type
    current_step_type: ConvStepType | None
    
    # The folders where the step implementations are located
    step_impl_folders: List[str]
        
    # The path where the test case will be saved by default
    test_case_path: str
    
    analysis_intention : LLMIntentionAnalysisResult | None
    
    ### Temporary data for the conversation
    #---------------------------------#
    user_input_lowercased: str # The raw user input that was received, lowercased and stripped of any extra spaces
    user_input_raw: str # The raw user input that was received, without any modifications
    prev_step_type: ConvStepType | None # The previous step type
    curr_step_reason: ConvStepType | None # The reason for the current step type. E.g, edit or not understanding. In these case we want to know the previous step type to come back to it
    #---------------------------------#
    
    def clear_clauses(self):
        self.clauses = {'given': [], 'when': [], 'then': []}

    def set_user_scenario(self, user_scenario: str):
        self.user_scenario = user_scenario

    def add_clause(self, clause_type: str, clause: str):
        clause_type = clause_type.lower().strip()

        # Remove the clause type from the clause and strip the clause of any extra spaces or quotes
        idx_prefix = clause.find(clause_type)
        if idx_prefix != -1:
            clause = clause[idx_prefix + len(clause_type):].strip('"').strip()

        self.clauses[clause_type].append(clause)

    def get_clauses(self, clause_type: str | None = None):
        return self.clauses

    def get_user_scenario(self):
        return self.user_scenario
    
    def set_current_step_type(self, step_type: ConvStepType):
        # Check if the conversation is diverging and the user is trying to edit a previous step or the LLM does not understand the user input
        is_diverging = step_type == ConvStepType.ASSISTANT_DOES_NOT_UNDERSTAND or \
                            step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT or \
                            step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT
           
        # When the conversation is diverging, we want to keep the previous step type
        if is_diverging:
            self.curr_step_reason = step_type            
            self.prev_step_type = self.current_step_type
        
        # Set the current step type
        self.current_step_type = step_type
        
    # This function checks if the conversation is diverging and the user is trying to edit a previous step or the LLM does not understand the user input
    # Returns True if the current graph could end, False otherwise
    # The graphs end currently when the user is trying to edit a previous step only
    def check_transition_to_next_step(self, next_step_type: ConvStepType) -> bool:
        # If there is any diverging reason, we want to go back to the previous step
        if self.curr_step_reason is not None:
            self.current_step_type = self.prev_step_type
            
            # Reset the diverging reason
            self.prev_step_type = None
            self.curr_step_reason = None
            
            if self.current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT or \
                self.current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT:
                return True
  
        return False  
    
    def get_bdd_step_from_current_state(self) -> GherkinStepType:
        if self.current_step_type == ConvStepType.GIVEN:
            return GherkinStepType.GIVEN
        elif self.current_step_type == ConvStepType.WHEN:
            return GherkinStepType.WHEN
        elif self.current_step_type == ConvStepType.THEN:
            return GherkinStepType.THEN
        else:
            # Well, we anot in a BDD step
            logging.error("Internal state error - we are not in a BDD step state currently")
            return None
        
    def get_current_step_type(self) -> ConvStepType:
        return self.current_step_type
    
    def get_current_step_message(self):        
        return self.analysis_intention.format_message(self.current_step_type)
    
        
        is_customized_message = self.current_step_type in [ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT, 
                                                           ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT, 
                                                           ConvStepType.ASK_FOR_USER_CONFIRMATION_MANAGEMENT_OP]
        
        if self.current_step_type == ConvStepType.ASSISTANT_DOES_NOT_UNDERSTAND:
            # TODO: Add a message for when the assistant does not understand the user input
            # Try to customize the message based on the analysis to let user know what the assistant does not understand
            pass 
        elif not is_customized_message:
            return assistant_message_map[self.current_step_type]
        else:
            # Customize the message based on the current step type and state
            customized_msg = None
            if self.current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT:
                customized_msg = self.analysis_intention.format_customized_message(self.current_step_type)
                AssistantFixedMessage.ask_for_user_confirmation_scenario_edit.format(what_to_edit=self.analysis_intention.SCENARIO_EDIT_TYPE)
            elif self.current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT:
                customized_msg = AssistantFixedMessage.ask_for_user_confirmation_bdd_edit.format(what_to_edit=self.analysis_intention.BDD_STEP_TYPE)
            elif self.current_step_type == ConvStepType.ASK_FOR_USER_CONFIRMATION_MANAGEMENT_OP:
                customized_msg = AssistantFixedMessage.ask_for_user_confirmation_management.format(operation=self.analysis_intention.intention, 
                                                                                                   path_of_loading=self.analysis_intention.temp_path)
                
            return customized_msg  

            ConvStepType.ASK_FOR_USER_CONFIRMATION_SCENARIO_EDIT: AssistantFixedMessage.ask_for_user_confirmation_scenario_edit,
            ConvStepType.ASK_FOR_USER_CONFIRMATION_BDD_EDIT: AssistantFixedMessage.ask_for_user_confirmation_bdd_edit,
            ConvStepType.ASK_FOR_USER_CONFIRMATION_MANAGEMENT_OP: AssistantFixedMessage.ask_for_user_confirmation_management,
    
        # TOODOOOOO
    
    def set_step_impl_folders(self, step_impl_folders: List[str]):
        self.step_impl_folders = step_impl_folders
        
    def get_step_impl_folders(self):
        return self.step_impl_folders
    
    def set_test_case_path(self, test_case_path: str):
        self.test_case_path = test_case_path
        
    def analyze_user_input(self):
        ############# Step 1: Understand the user input in the context #############
        # Will output results in the analysis_intention variable
        
        # Assess LLM's input at some point depending on the current step
        print(f"llm_check - current_step_type: {self.get_current_step_type()} - prev_step_type: {self.prev_step_type}")

        # so called decision making internal by the assistant, depending on user input and history
        # TODO - Here we need to call LLM and get the INDENTION + REFLECT on user history + the user input => return the response, ask for confirmation and move to the next step
        
        # TODO: check again here the tools functions and they would fit in 
        #---------------------------------------------------
        
        """ 
        Example user input:
        I want to edit the path to the scenario output
        I want to modify the scenario description
        I want to edit / modify the Given / When / Then step
        
        Response from LLM:
        """
        
        self.analysis_intention = None # LLM does not understand the user input
        
        ANALYSIS_MODE_USING_LLM = False # We can use LLM to analyze the user input, or rule-based system
        if not ANALYSIS_MODE_USING_LLM:    
            # We handle the decision using rule-based for now
            
            # Rule 1: If the user seems to be to edit/change/modify/add the scenario, we should understand the user what they want to edit
            edit_intentions = ['edit', 'change', 'modify', 'add']        
            edit_intention_a_scenario = ['scenario']
            edit_intention_a_step = ['step', 'bdd']        
            edit_intention_which = edit_intention_a_scenario + edit_intention_a_step
            
            # Test if the user input contains the edit intention and the what to edit, the entity
            is_edit_intention =  any_word_in_sentence(edit_intentions, self.user_input_lowercased)  \
                                    and any_word_in_sentence(edit_intention_which, self.user_input_lowercased)
                    
            if is_edit_intention:
                # Now the decision splits in two, is it the scenario or the bdd step that the user wants to edit ? 
                # This is fixed by the previous step for now but we can also use the user input to decide            
                is_output_type =  any_word_in_sentence(['path', 'output', 'out', 'save'], self.user_input_lowercased)
                is_code_type = any_word_in_sentence(['code'] in self.user_input_lowercased)
                is_description_type = any_word_in_sentence(['description'], self.user_input_lowercased)
                    
                # Rule 1.1: If the user seems to be to edit the scenario, we should ask the user what they want to edit
                if 'scenario' in self.user_input_lowercased:                                        
                    if is_output_type or is_code_type or is_description_type:                    
                        #could_end = state.check_transition_to_next_step(ConvStepType.SCENARIO_DEF_EDIT_SELECTION)
                        # assert could_end, "llm_analysis_node: SCENARIO_DEF_EDIT_SELECTION could not be reached when user input is edit"
                        
                        scenario_edit_type = ConvStepType.SCENARIO_DEF_PART2_SCENARIO_OUTPUTPATH if is_output_type else \
                                                                            ConvStepType.SCENARIO_DEF_PART3_CODE_PATH if is_code_type else \
                                                                            ConvStepType.SCENARIO_DEF_PART1_SCENARIO_DEF if is_description_type else None
                        
                        self.analysis_intention = LLMIntentionAnalysisResult(LLMAnalysisIntention.LLM_INTENTION_EDIT_SCENARIO,
                                                                            SCENARIO_EDIT_TYPE=scenario_edit_type)
                        
                    else:
                        self.analysis_intention = LLMIntentionAnalysisResult(LLMAnalysisIntention.LLM_INTENTION_EDIT_SCENARIO,
                                                                            SCENARIO_EDIT_TYPE=None)
                                                                            
                elif any_word_in_sentence(['step', 'bdd'], self.user_input_lowercased):
                    steps_in_input = (['given', 'when', 'then'], self.user_input_lowercased)
                    if any(steps_in_input):
                        #could_end = state.check_transition_to_next_step(ConvStepType.BDD_STEP_EDIT_SELECTION)
                        #assert could_end, "llm_analysis_node: ADD_EDIT_BDD_STEP could not be reached when user input is edit"
                                            
                        bdd_step_type = GherkinStepType.GIVEN if 'given' in self.user_input_lowercased else \
                                                        GherkinStepType.WHEN if 'when' in self.user_input_lowercased else \
                                                        GherkinStepType.THEN if 'then' in self.user_input_lowercased else None
                        
                        self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_EDIT_BDD_STEP,
                                                                            BDD_STEP_TYPE=bdd_step_type)
                        
                    else:
                        # Mark that the step is missing in the user input
                        self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_EDIT_BDD_STEP,
                                                                            BDD_STEP_TYPE=None)
                        
                elif any_word_in_sentence(['save', 'load', 'discard'], self.user_input_lowercased):
                    is_save = any_word_in_sentence(['save'], self.user_input_lowercased)
                    is_load = any_word_in_sentence(['load'], self.user_input_lowercased)
                    is_discard = any_word_in_sentence(['discard'], self.user_input_lowercased)
                    
                    if is_save:
                        self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_SAVE_SCENARIO)
                    elif is_load:
                        if any_word_in_sentence(['path', 'output'], self.user_input_lowercased):
                            self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_LOAD_SCENARIO)
                            
                            # Find the path in the user input
                            path_start = min([self.user_input_lowercased.find(x) for x in ['path', 'output']])
                            # Path ends at the first space or tab or newline
                            path_prefix = self.user_input_lowercased[path_start:].strip()
                            
                            path_end_indices = [self.user_input_lowercased[path_prefix].find(x) for x in [' ', '\t', '\n']]
                            path_end_index = min(path_end_indices) if len(path_end_indices) > 0 else len(path_prefix)
                            
                            presumed_path = path_prefix[:path_end_index]
                            
                            # Check if the path is valid
                            if os.path.exists(presumed_path):
                                self.analysis_intention.temp_path = presumed_path
                            else:
                                self.analysis_intention.temp_path = None
                            
                        self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_LOAD_SCENARIO)
                    elif is_discard:
                        self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_DISCARD_SCENARIO)
                    else:
                        self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_UNKNOWN)
            else:
                # Check if user is giving the step definition according to the flowchart
                if self.user_input_lowercased.startswith('ok') or self.user_input_lowercased.startswith('yes') or self.user_input_lowercased.startswith('here'):
                    # Get rid of the user prefix message
                    user_prefix_msg = min([self.user_input_raw.find(x) for x in ['\n', ':']])
                    
                    self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_ADD_BDD_STEP,
                                                                        BDD_STEP_TYPE=self.get_bdd_step_from_current_state(),
                                                                        user_input=self.user_input_raw[user_prefix_msg:])
                else: 
                    self.analysis_intention = LLMIntentionAnalysisResult(intention= LLMAnalysisIntention.LLM_INTENTION_ADD_BDD_STEP,
                                                                        BDD_STEP_TYPE=self.get_bdd_step_from_current_state(),
                                                                        user_input=self.user_input_raw)
                            
            if self.analysis_intention is None:
                self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_UNKNOWN)

        else:
            # Call LLM to get the intention - i think we already have some code for this 
            # Not implemented yet
            #return None 
            #assert False, "llm_analysis_node: LLM is not implemented yet"
            self.analysis_intention = LLMIntentionAnalysisResult(intention=LLMAnalysisIntention.LLM_INTENTION_UNKNOWN) 
        
        
        ########## Step 2: Final analysis and decision making ##########
        # We do some further processing in the following cases:
        # Skip for now, not needed for the current implementation
        
        #---------------------------------------------------