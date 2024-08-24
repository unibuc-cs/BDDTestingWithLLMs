from typing import Tuple

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
def fill_regex_with_dictvalues(step_regex: str, replacements: dict) -> Tuple[bool, str]:
    stack = []
    slen = len(step_regex)

    filled_str = ""
    last_unmatched = 0

    error_msg = ""

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
                    error_msg  += f"I am unable to find the value for key {key_to_replace} in the matched parameters dictionary. \n"
                    # Continue, but log the error maybe user can see it
                filled_str += step_regex[last_unmatched:i] + replacements[key_to_replace]

                last_unmatched = i + closing_index + 1

    filled_str += step_regex[last_unmatched:]

    if error_msg == "":
        return True, filled_str
    else:
        return False, error_msg

