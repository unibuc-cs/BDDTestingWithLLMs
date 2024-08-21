import sys
sys.path.append("./car-behave-master")
from car import Car



def before_scenario(context, scenario):
    context.car = Car()