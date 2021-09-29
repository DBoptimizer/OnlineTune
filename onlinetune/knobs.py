import json
import time
import bisect
from .utils import logger
import numpy as np

ts = int(time.time())
logger = logger.get_logger('onlinetune', 'log/train_ddpg_{}.log'.format(ts))
KNOB_DETAILS = None


def gen_continuous(action):
    knobs = {}

    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]

        knob_type = value['type']

        if knob_type == 'integer':
            min_val, max_val = value['min'], value['max']
            delta = int((max_val - min_val) * action[idx])
            eval_value = min_val + delta 
            eval_value = max(eval_value, min_val)
            if value.get('stride'):
                all_vals = np.arange(min_val, max_val, value['stride'])
                indx = bisect.bisect_left(all_vals, eval_value)
                if indx == len(all_vals): indx -= 1
                eval_value = all_vals[indx]

            knobs[name] = eval_value
        if knob_type == 'float':
            min_val, max_val = value['min'], value['max']
            delta = (max_val - min_val) * action[idx]
            eval_value = min_val + delta
            eval_value = max(eval_value, min_val)
            knobs[name] = eval_value
        elif knob_type == 'enum':
            enum_size = len(value['enum_values'])
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['enum_values'][enum_index]
            knobs[name] = eval_value
        elif knob_type == 'combination':
            enum_size = len(value['combination_values'])
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = value['combination_values'][enum_index]
            knobs_names = name.strip().split('|')
            knobs_value = eval_value.strip().split('|')
            for k, knob_name_tmp in enumerate(knobs_names):
                knobs[knob_name_tmp] = knobs_value[k]


    return knobs


def initialize_knobs(knobs_config, num):
    global KNOBS
    global KNOB_DETAILS
    if num == -1:
        f = open(knobs_config)
        KNOB_DETAILS = json.load(f)
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
    else:
        f = open(knobs_config)
        knob_tmp = json.load(f)
        i = 0
        KNOB_DETAILS = {}
        while i < num:
            key = list(knob_tmp.keys())[i]
            KNOB_DETAILS[key] = knob_tmp[key]
            i = i + 1
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
    return KNOB_DETAILS


def get_default_knobs():
    default_knobs = {}
    for name, value in KNOB_DETAILS.items():
        if not value['type'] == "combination":
            default_knobs[name] = value['default']
        else:
            knobL = name.strip().split('|')
            valueL = value['default'].strip().split('|')
            for i in range(0, len(knobL)):
                default_knobs[knobL[i]] = int(valueL[i])
    return default_knobs


def get_knob_details(knobs_config):
    initialize_knobs(knobs_config, -1)
    return KNOB_DETAILS


def knob2action(knob):
    actionL = []
    for idx in range(len(KNOBS)):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]
        knob_type = value['type']
        if knob_type == "enum":
            enum_size = len(value['enum_values'])
            action = value['enum_values'].index(knob[name]) / enum_size
        else:
            min_val, max_val = value['min'], value['max']
            action = (knob[name] - min_val) / (max_val - min_val)

        actionL.append(action)

    return np.array(actionL)

