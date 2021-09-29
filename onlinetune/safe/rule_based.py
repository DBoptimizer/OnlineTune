import os
import pdb
import copy
import numpy as np
from autotune.knobs import logger
from autotune.knobs import gen_continuous, knob2action
mysqltuner_path="/data2/MySQLTuner-perl/mysqltuner.pl"

class rule():
    def __init__(self, knobs_detail):
        self.knobs_detail = knobs_detail
        self.current_log_num = 0
        self.rule_against = {}
        self.rule_against_safe = {}
        self.current_rule_against = {}
        self.explore_ratio = 0.3
        self.relax_threshold = {}
        self.block_threshold = {}
        for k in knobs_detail.keys():
            self.relax_threshold[k] = 3
            self.block_threshold[k] = 3
            if k == 'innodb_thread_concurrency' or  k == 'max_join_size':
                self.block_threshold[k] = 6

        self.relax_list = []
        self.relax_list_dba = []
        self.template_suggest = copy.deepcopy(self.knobs_detail)
        self.dba_rules()


    def add_dba_rules(self, knob):
        for k in self.dba_rules.keys():
            if self.dba_rules[k]['operate']  == 'min':
                if self.dba_rules[k]['value'] > self.template_suggest[k]['min']:
                    self.template_suggest[k]['min'] = self.dba_rules[k]['value']
            elif self.dba_rules[k]['operate'] == 'max':
                if self.dba_rules[k]['value'] < self.template_suggest[k]['max']:
                    self.template_suggest[k]['max'] = self.dba_rules[k]['value']
            elif self.dba_rules[k]['operate'] == 'multiple_min':
                knobs_reference =  self.dba_rules[k]['reference']
                knobs_value = knob[knobs_reference] * self.dba_rules[k]['value']
                if knobs_value > self.template_suggest[k]['min']:
                    self.template_suggest[k]['min'] = knobs_value

    def relax_dba_rules(self, k):
        self.relax_list_dba.append(k)
        self.dba_rules[k]['past_value'].append(self.dba_rules[k]['value'])


        if self.dba_rules[k]['operate'] == 'multiple_min':
            self.dba_rules[k]['value'] = self.dba_rules[k]['value_change']
            if self.dba_rules[k]['value'] < 1:
                self.dba_rules[k]['value'] = 1

        else:
            self.dba_rules[k]['value'] = self.dba_rules[k]['value_change']

        logger.info("rule value for {} is relaxed to {}".format(k, self.dba_rules[k]['value']))


    def restrict_dba_rules(self, k):
        self.relax_list_dba.remove(k)
        if len(self.relax_list_dba['past_value']) > 0:
            back_value = self.relax_list_dba['past_value'].pop()
            self.dba_rules[k]['value'] = back_value
            logger.info("rule value for {} is restricted to {}".format(k, back_value))
        else:
            logger.info("wrong logic")


    def dba_rules(self):
        self.dba_rules = {}
        self.dba_rules['innodb_thread_concurrency'] = {}
        self.dba_rules['innodb_thread_concurrency']['operate'] = 'min'
        self.dba_rules['innodb_thread_concurrency']['value'] = 5
        self.dba_rules['innodb_thread_concurrency']['value_change'] = 5
        self.dba_rules['innodb_thread_concurrency']['past_value'] = []

        self.dba_rules['max_join_size'] = {}
        self.dba_rules['max_join_size']['operate'] = 'min'
        self.dba_rules['max_join_size']['value'] = int(self.knobs_detail['max_join_size']['default'] / 10)
        self.dba_rules['max_join_size']['value_change'] = int(self.knobs_detail['max_join_size']['default'] / 10)
        self.dba_rules['max_join_size']['past_value'] = []

        self.dba_rules['key_buffer_size'] = {}
        self.dba_rules['key_buffer_size']['operate'] = 'min'
        self.dba_rules['key_buffer_size']['value'] = self.knobs_detail['key_buffer_size']['default']
        self.dba_rules['key_buffer_size']['value_change'] = self.knobs_detail['key_buffer_size']['default']
        self.dba_rules['key_buffer_size']['past_value'] = []

        self.dba_rules['innodb_spin_wait_delay'] = {}
        self.dba_rules['innodb_spin_wait_delay']['operate'] = 'max'
        self.dba_rules['innodb_spin_wait_delay']['value'] = 10
        self.dba_rules['innodb_spin_wait_delay']['value_change'] = 10
        self.dba_rules['innodb_spin_wait_delay']['past_value'] = []

        self.dba_rules['table_open_cache'] = {}
        self.dba_rules['table_open_cache']['operate'] = 'min'
        self.dba_rules['table_open_cache']['value'] = 32
        self.dba_rules['table_open_cache']['value_change'] = 32
        self.dba_rules['table_open_cache']['past_value'] = []

        self.dba_rules['innodb_io_capacity_max'] = {}
        self.dba_rules['innodb_io_capacity_max']['operate'] = 'multiple_min'
        self.dba_rules['innodb_io_capacity_max']['reference'] = 'innodb_io_capacity'
        self.dba_rules['innodb_io_capacity_max']['value'] = 1.5
        self.dba_rules['innodb_io_capacity_max']['value_change'] = 1.5
        self.dba_rules['innodb_io_capacity_max']['past_value'] = []




    def get_number(self, value):
        if ',' in value:
            value = value[:value.find(',')]

        if 'B' in value:
            value = eval(value[:-1])
        elif 'K' in value:
            value = eval(value[:-1]) * 1024
        elif 'M' in value:
            value = eval(value[:-1]) * 1024*1024
        elif 'G' in value:
            value = eval(value[:-1]) * 1024*1024*1024
        return int(value)

    def get_suggest_value(self, knob, line):
        markL = ['< ', '> ', '>= ', '<= ', '=']
        for mark in markL:
            if mark in line:
                line = line[ line.find(mark) + len(mark):]
                value = line.split()[0].strip(')')
                value = self.get_number(value)
                if mark == '< ' or mark == '<= ' or mark == '=':
                    if value * 0.99 < self.template_suggest[knob]['max']:
                        self.template_suggest[knob]['max'] = int(value  * 0.99)
                elif mark == '> ' or mark == '>= ' or mark == '=':
                    if value * 1.01 > self.template_suggest[knob]['min']:
                        self.template_suggest[knob]['min'] = int(value * 1.01)
        if '(start at 4)' in line:
            self.template_suggest[knob]['min'] = 4







    def suggest(self, log_num):
        self.template_suggest = copy.deepcopy(self.knobs_detail)
        self.current_log_num = log_num
        suggest_flag = False
        cmd = "perl {} --host 127.0.0.1  --skippassword " \
              "--notbstat     --nocolstat   --noinfo  --silent  " \
              " --outputfile=mysql_tuner/{}.txt".\
            format(mysqltuner_path, self.current_log_num)

        os.system(cmd)
        f = open("mysql_tuner/{}.txt".format(self.current_log_num))
        lines = f.readlines()

        if "Variables to adjust:\n" in lines:
            begin_index = lines.index("Variables to adjust:\n")
            lines = lines[begin_index:]
            sugestL = []
            for line in lines:
                if 'innodb_buffer_pool_siz' in line or 'innodb_buffer_pool_instances' in line\
                        or 'thread_cache_size' in line:
                    continue
                print (line.strip())
                logger.info(line.strip())
                sugestL.append(line)

            for line in sugestL:
                suggest_flag = True
                for k in self.knobs_detail.keys():
                    if k in line :
                        if k in self.relax_list:
                            self.current_rule_against[k] = True
                            logger.info("rule for {} has been relaxed, not changing".format(k))
                            continue
                        self.get_suggest_value(k, line)


        return suggest_flag, self.template_suggest



    def check_safe_action(self, action, no_explore=False):
        action_num = len(self.knobs_detail.keys())
        action = action.flatten()
        add_context = False
        if action.shape[0] > action_num:
            context = action[action_num:].copy()
            action = action[:action_num]
            add_context = True

        knobs = gen_continuous(action)
        knobs_new = self.check_safe(knobs, no_explore)
        if knobs == knobs_new:
            if add_context:
                action = np.hstack((action, context))
            return action
        else:
            action_new = knob2action(knobs_new)
            if add_context:
                action_new = np.hstack((action_new, context))

            return action_new



    def check_safe(self, knobs, no_explore = False):
        knobs = copy.deepcopy(knobs)
        self.add_dba_rules(knobs)
        self.template_suggest['innodb_io_capacity_max']['min'] = knobs['innodb_io_capacity']
        for k in self.template_suggest.keys():
            if not self.template_suggest[k]['type'] == 'integer':
                continue
            if knobs[k] < self.template_suggest[k]['min']:
                if k == 'innodb_thread_concurrency' and knobs[k] == 0:
                    continue
                logger.info("rule for {} > {} is against".format(k, self.template_suggest[k]['min']))
                self.rule_against[k] = self.rule_against.get(k, 0) + 1
                logger.info("rule for {} is against {} times".format(k,  self.rule_against[k]))


                if no_explore \
                    or np.random.rand() > self.explore_ratio \
                    or self.rule_against[k] < self.block_threshold[k]\
                        or (k in self.dba_rules.keys() and knobs[k] < self.template_suggest[k]['min'] * 0.5): #change according to advice
                    logger.info("rule based change {}: {} -> {}".format(k, knobs[k], self.template_suggest[k]['min']))
                    knobs[k] = self.template_suggest[k]['min']
                else:
                    self.current_rule_against[k] = True
                    no_explore = True
                    if k in self.dba_rules.keys() and knobs[k] < self.dba_rules[k]['value_change']:
                        self.dba_rules[k]['value_change'] = knobs[k]

                    logger.info("explore {} is not changed".format(k))

            if  knobs[k] > self.template_suggest[k]['max']:
                logger.info("rule for {} < {} is against".format(k, self.template_suggest[k]['max']))
                self.rule_against[k] = self.rule_against.get(k, 0) + 1
                logger.info("rule for {} is against {} times".format(k, self.rule_against[k]))

                if no_explore \
                    or np.random.rand() > self.explore_ratio \
                    or self.rule_against[k] < self.block_threshold[k]\
                        or (k in self.dba_rules.keys() and knobs[k] > self.template_suggest[k]['max'] * 1.5):  #change according to advice
                    logger.info("rule based change {}: {} -> {}".format(k, knobs[k], self.template_suggest[k]['max']))
                    knobs[k] = self.template_suggest[k]['max']
                else:
                    self.current_rule_against[k] = True
                    no_explore = True
                    self.rule_against[k] = 0
                    if k in self.dba_rules.keys() and knobs[k] > self.dba_rules[k]['value_change'] :
                        self.dba_rules[k]['value_change'] = knobs[k]

                    logger.info("explore {} is not changed".format(k))


        return knobs


    def feedback(self, safe):
        if safe == True:
            for k in self.current_rule_against:
                self.rule_against_safe[k] = self.rule_against_safe.get(k, 0) + 1
                logger.info("rule for {} is against but safe {} times".format(k, self.rule_against_safe[k]))

        if safe == False:
            for k in self.current_rule_against:
                self.rule_against_safe[k] = 0
                self.rule_against[k] = 0
                self.relax_threshold[k] = self.relax_threshold[k] + 1
                if k in self.dba_rules.keys():
                    self.dba_rules[k]['value_change'] =  self.dba_rules[k]['value']
                logger.info("rule for {} is against and unsafe, its relax_threshold change to {}".format(k, self.relax_threshold[k]))



        for k in self.rule_against_safe.keys():
            if self.rule_against_safe[k] > self.relax_threshold[k]:
                logger.info(
                    "Accumulative safe against > {}, the rule for {} is relaxed".format(self.relax_threshold[k], k))
                if not k in self.relax_list:
                    self.relax_list.append(k)
                    self.rule_against[k] = 0
                if k in self.dba_rules.keys():
                    self.rule_against[k] = 0
                    self.relax_dba_rules(k)

        for k in self.relax_list:
            if self.rule_against_safe[k] == 0:
                self.rule_against[k] = 0
                logger.info("Against rule leads to unsafe knobs, the rule for {} is restored".format(k))
                self.relax_list.remove(k)

        for k in self.relax_list_dba:
            if self.rule_against_safe[k] == 0:
                self.restrict_dba_rules(k)


        self.current_rule_against = {}














