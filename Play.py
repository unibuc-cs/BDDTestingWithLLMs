import os
import re


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

# Example usage
# directory = './car-behave-master/features/steps'
# clauses = extract_clauses(directory)
# print(clauses)


import json

text = """{
  "found": true,
  "step_found": '"the car has (?P<engine_power>\\\\d+) kw, weights (?P<weight>\\\\d+) kg, has a drag coefficient of (?P<drag>[\\\\.\\\\d]+)"'
}"""

json.loads(text)