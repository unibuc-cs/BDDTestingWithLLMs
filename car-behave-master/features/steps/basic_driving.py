from behave import *
from hamcrest import assert_that, close_to, greater_than, less_than, equal_to

use_step_matcher('re')


@given("the car has (?P<engine_power>\d+) kw, weighs (?P<weight>\d+) kg, has a drag coefficient of (?P<drag>[\.\d]+)")
def step_impl(context, engine_power, weight, drag):
    context.car.engine_power = float(engine_power)
    context.car.weight = float(weight)
    context.car.drag = float(drag)


@given("a frontal area of (?P<area>.+) m\^2")
def step_impl(context, area):
    context.car.frontal_area = float(area)

@given("that the car is moving at (?P<speed>\d+) m/s")
def step_impl(context, speed):
    context.car.speed = float(speed)


@given("that the car's heading is (?P<heading>\d+) deg")
def step_impl(context, heading):
    context.car.heading = float(heading)

def helper_init_road_inclination(context, road_inclination:str):
    if road_inclination == 'straight road':
        context.car.road_inclination = 0.0
    elif road_inclination == 'hill':
        context.car.road_inclination = 0.1
    else:
        context.car.road_inclination = -0.1


@given("the inclination of the road is straight")
def step_impl(context):
    helper_init_road_inclination(context, 'straight road')

@given("the inclination of the road is hill")
def step_impl(context):
    helper_init_road_inclination(context, 'hill')

@given("the inclination of the road is downhill")
def step_impl(context):
    helper_init_road_inclination(context, 'downhill')

@given("the inclination of the road is (?P<road_inclination>.*)")
def step_impl(context, road_inclination):
    helper_init_road_inclination(context, road_inclination)


def helper_init_road_condition(context, road_condition:str):
    if road_condition == 'dry':
        context.car.road_condition = 1.0
    elif road_condition == 'wet':
        context.car.road_condition = 0.8
    else:
        context.car.road_condition = 0.3

@given("the car in on dry road")
def step_impl(context):
    helper_init_road_condition(context, 'dry')

@given("the car in on wet road")
def step_impl(context):
    helper_init_road_condition(context, 'wet')

@given("the car in on icy road")
def step_impl(context):
    helper_init_road_condition(context, 'icy')

@given("the car in on (?P<road_condition>.*) road")
def step_impl(context, road_condition):
    helper_init_road_condition(context, road_condition)

@given("the car has (?P<brake_force>\d+) N of braking force")
def step_impl(context, brake_force):
    context.car.brake_force = float(brake_force)

@given("the car has (?P<yaw_rate>\d+) deg/s of yaw rate")
def step_impl(context, yaw_rate):
    context.car.yaw_rate = float(yaw_rate)


#########
@when("I accelerate to (?P<speed>\d+) km/h")
def step_impl(context, speed):
    speed_in_ms = float(speed) / 3.6
    context.car.set_power(100)
    while context.car.speed < speed_in_ms:
        context.car.simulate(0.1)

@when("I brake at (?P<brake_force>\d+)% force")
def step_impl(context, brake_force):
    context.car.set_brake(brake_force)

@when("I turn (?P<direction>left|right) at a yaw rate of (?P<rate>\d+) deg/sec for (?P<duration>\d+) seconds")
def step_impl(context, direction, rate, duration):
    if direction == 'left':
        context.car.turn_left(rate)
    else:
        context.car.turn_right(rate)

    context.car.simulate(duration)

@then("the time should be within (?P<precision>[\d\.]+)s of (?P<time>[\d\.]+)s")
def step_impl(context, precision, time):
    assert_that(context.car.time, close_to(float(time), float(precision)))


@step("(?P<seconds>\d+) seconds? pass(?:es)?")
def step_impl(context, seconds):
    context.car.simulate(seconds)


@then("I should have traveled less than (?P<distance>\d+) meters")
def step_impl(context, distance):
    assert_that(context.car.odometer, less_than(float(distance)))


@then("the car's heading should be (?P<heading>\d+) deg")
def step_impl(context, heading):
    assert_that(context.car.heading, equal_to(float(heading)))

