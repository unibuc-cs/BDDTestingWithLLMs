from typing import Tuple, Dict
import re
import os
import spacy
import Levenshtein
# Load the language model for similarity detection
en_tokens_similarity_model = spacy.load('en_core_web_md')


# Given a directory, extract Gherkin clauses by type
def extract_clauses(directory):
    clauses = {'given': [], 'when': [], 'then': []}
    pattern = re.compile(r'@(|given|then|when|And|But)\((.*)\)')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    matches = pattern.findall(content)
                    for match in matches:
                        clauses[match[0]].append(match[1])
    return clauses


# Given a string, find the closing index of a matching parenthesis starting as open_pos
def find_matching_parenthesis(s: str, open_pos: int) -> int:
    stack = []
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                if start == open_pos:
                    return i
            else:
                return -1 # Incorrect, no matching opening parenthesis
    return -1  # Return -1 if no matching parenthesis is found


# Given a string with a regex and a dictionary of replacements, fill the regex with the values from the dictionary
# The regex should be in a text that contain named groups, e.g., (?P<key>value)
# The dictionary should have the keys as the named groups, and the values as the replacements
# The function will return a tuple with a boolean indicating if the operation was successful and the filled string
def fill_regex_with_dictvalues(step_regex: str, replacements: dict) -> Tuple[bool, str, str]:
    stack = []
    slen = len(step_regex)

    filled_str = ""
    last_unmatched = 0

    error_msg = ""
    warning_msg = ""

    # In case that the model is not able to provide exactly the same named groups as the regex, we can backup the variables found and compare in the end to the closest match
    replacement_values : Dict[str, str] = {}
    used_replacement = set()
    for i, char in enumerate(step_regex):
        if char == '(':
            remain_sz = slen - i
            if remain_sz > 3 and step_regex[i + 1] == '?' and step_regex[i + 2] == 'P' and step_regex[i + 3] == '<':
                closing_index = find_matching_parenthesis(step_regex[i:], 0)
                if closing_index == -1:
                    error_msg += f"Incorrect, no matching closing parenthesis when finding the closing parenthesis for a named group, at position {i}, string {step_regex[i:]}. \n"
                    break # Break, as we can't continue without a matching closing parenthesis
                group = step_regex[i:i + closing_index + 1]

                key_to_replace = group[4:group.index('>')]

                if key_to_replace not in replacements:
                    replacement_values[key_to_replace] = "" # Add the key to the backup variables
                else:
                    replacement_values[key_to_replace] = replacements[key_to_replace]
                    used_replacement.add(key_to_replace)

                filled_str += step_regex[last_unmatched:i] + "{" + str(key_to_replace) + "}"

                last_unmatched = i + closing_index + 1

    filled_str += step_regex[last_unmatched:]


    # Test if two strings are similar
    def similar(key1, key2):
        # Lowercase the keys
        key1 = key1.lower()
        key2 = key2.lower()

        # Check if the keys are the same or if one is a substring of the other
        if key1 == key2:
            return True
        if key1 in key2:
            return True
        if key2 in key1:
            return True

        # TODO: compute the highest similarity between all to all - bipartite
        # Calculate similarity semantically
        token1 = en_tokens_similarity_model(key1)
        token2 = en_tokens_similarity_model(key2)
        similarity = token1.similarity(token2)
        if similarity > 0.6:
            return True

        # compute the levenstein distance between the two keys
        # if the distance is less than 2, we consider them similar
        if Levenshtein.ratio(key1, key2) > 0.6:
            return True

        # Common words similarity - maybe per projects?
        similar_words = {"brake": "braking", "force": "braking", "yaw": "turn", "rate": "turn", "seconds":
            "time", "pass": "time", "travel": "distance", "meters": "distance", "within": "close_to", "s":
            "close_to", "close": "close_to", "to": "close_to", "of": "close_to", "should": "assert_that",
                         "assert": "assert_that", "that": "assert_that", "the": "assert_that", "be": "assert_that",
        "mass:": "mass", "kg": "mass", "drag": "drag_coefficient", "coefficient": "drag_coefficient",
        "mass": "weight", "power": "engine_power", "kw": "engine_power", "weights": "weight", "weight": "weight",
        "has": "is", "a": "is", "deg": "degrees", "deg/s": "degrees", "deg/sec": "degrees"}
        if key1 in similar_words and similar_words[key1] == key2 or \
            key2 in similar_words and similar_words[key2] == key1:
            return True

        return False

    # Now check if there are any missing keys
    first_time_err = True
    for keyToReplace in replacement_values:
        # If the variable does not have a value, we need to find a similar key
        if replacement_values[keyToReplace] == "":
            found = False
            for replacement_key in replacements:
                if replacement_key not in used_replacement:
                    if similar(replacement_key, keyToReplace):
                        found = True
                        #filled_str = filled_str.replace(f"{{key_to_replace}}", f"{replacements[replacement_key]}")
                        warning_msg += f"\n I will replace key {keyToReplace} to key {replacement_key} - the most similar one."
                        replacement_values[keyToReplace] = replacements[replacement_key]
                        used_replacement.add(replacement_key)
                        break

            if not found:
                error_msg += f"\nCould not find a similar key {keyToReplace}"
                if first_time_err:
                    first_time_err = False
                    error_msg += f" in the replacements available: {replacements}. \n"

        else:
            # Replace the key with the value
            pass
            #filled_str = filled_str.replace(f"{key_to_replace}", f"{replacement_values[keyToReplace]}")

    filled_str = filled_str.format(**replacement_values)

    if error_msg == "":
        return True, filled_str, warning_msg
    else:
        return False, error_msg, warning_msg

