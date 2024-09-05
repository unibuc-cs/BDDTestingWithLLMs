INFERENCE_PROMPT_TEMPLATE = """
Generate {N} unit tests for class name {CLASS_NAME} considering the following requirements:
The target class import directives and source code is: {CLASS_CORPUS}
The available mock classes interfaces are {MOCK_CLASSES} and the available mock classes source code is {MOCK_CLASSES_CORPUS}
All generated tests should be enclosed within <UnitGen> and </UnitGen> tags. Consider the followings:
Import required packages for types used.
Consider the mocks provided by input. If not available generate samples for each as required.
Always add assert statements at the end of the test, at least one for each unit test.
"""

# The target class signature is: {CLASS_SIGNATURE} - taken out from above prompt

import langchain_core
import langchain_core.prompts

from langchain_core.prompts import PromptTemplate

INFERENCE_PROMPT_TEMPLATE = PromptTemplate(template=INFERENCE_PROMPT_TEMPLATE,
                                  input_variables=["N",
                                                   "CLASS_NAME",
                                                   "CLASS_SIGNATURE",
                                                   "CLASS_CORPUS",
                                                   "MOCK_CLASSES",
                                                   "MOCK_CLASSES_CORPUS",
                                                   ])

USER_HF_TOKEN = "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"