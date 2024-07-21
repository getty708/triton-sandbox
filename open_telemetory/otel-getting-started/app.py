import logging
from random import randint

from flask import Flask, request
from opentelemetry import metrics, trace

# Acquire a tracer
tracer = trace.get_tracer("diceroller.tracer")
# Acquire a meter.
meter = metrics.get_meter("diceroller.meter")

roll_counter = meter.create_counter(
    "dice.roll",
    description="The number of rolls by the roll value",
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/rolldice")
def roll_dice():
    with tracer.start_as_current_span("roll") as roll_span:
        player = request.args.get("player", default=None, type=str)
        result = str(roll())
        roll_span.set_attribute("roll.value", result)
        roll_counter.add(1, {"roll.value": result})
        if player:
            logger.warn("{} is rolling the dice: {}", player, result)
        else:
            logger.warn("Anonymous player is rolling the dice: %s", result)
        return result


def roll():
    return randint(1, 6)
