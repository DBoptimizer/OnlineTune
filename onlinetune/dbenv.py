import os
import time
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from .dbconnector import MysqlConnector
from .knobs import logger
from .utils.parser import  parse_oltpbench,  parse_job
from .knobs import initialize_knobs, get_default_knobs, gen_continuous
try:
  import xml.etree.cElementTree as et
except ImportError:
  import xml.etree.ElementTree as et
from .utils.dynamic_worklod import gen_context
from .safe.rule_based import rule
from onlinetune.autoencoder import TextAutoencoder
from onlinetune.utils import encoder_utils


class DBEnv(ABC):
    def __init__(self, workload):
        self.score = 0.
        self.steps = 0
        self.terminate = False
        self.workload = workload

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self, knobs, episode, step):
        pass

    @abstractmethod
    def terminate(self):
        return False


class MySQLEnv(DBEnv):
    def __init__(self,
                 workload,
                 knobs_config,
                 log_path='',
                 threads=8,
                 host='localhost',
                 port=3392,
                 user='root',
                 passwd='',
                 dbname='tpcc',
                 rds_mode=True,
                 oltpbench_config_xml='',
                 pid=9999,
                 knob_num=-1
                 ):
        super().__init__(workload)
        self.knobs_config = knobs_config
        self.mysqld = os.environ.get('MYSQLD')
        self.mycnf = os.environ.get('MYCNF')
        self.sock = os.environ.get('MYSOCK')
        if not self.mysqld:
            logger.error('You should set MYSQLD env var before running the code.')
        if not self.mycnf:
            logger.error('You should set MYCNF env var before running the code.')
        if not self.sock:
            logger.error('You should set MYSOCK env var before running the code.')
        self.workload = workload
        self.log_path = log_path
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.threads = threads
        self.knobs_detail = initialize_knobs(knobs_config, knob_num)
        self.default_knobs = get_default_knobs()
        self.rds_mode = rds_mode
        self.oltpbench_config_xml = oltpbench_config_xml
        self.step_count = 0
        self.connect_sucess = True
        self.pid = pid
        self.generate_time()
        self.log_file = 0
        self.workload_change_count = 0
        self.rule = rule(self.knobs_detail)
        self.knobs_rule = self.default_knobs
        self.wd = encoder_utils.WordDictionary("../Workload2Vec/LSTM/model/vocabulary.txt")
        self.sess = tf.InteractiveSession()
        self.encoder = TextAutoencoder.load("../Workload2Vec/LSTM/model", self.sess)
        self.rule_based = True
        self.dynamic_rate = False
        self.general_log_count = 0


    def generate_time(self):
        global BENCHMARK_RUNNING_TIME
        global BENCHMARK_WARMING_TIME
        global TIMEOUT
        global RESTART_FREQUENCY
        if self.workload['name'] == 'job':
            BENCHMARK_RUNNING_TIME = 180
            BENCHMARK_WARMING_TIME = 0
            TIMEOUT = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME
        if self.workload['name'] == 'oltpbench':
            BENCHMARK_RUNNING_TIME = 170
            BENCHMARK_WARMING_TIME = 10
            TIMEOUT = BENCHMARK_RUNNING_TIME + BENCHMARK_WARMING_TIME

    def generate_knobs(action):
            return gen_continuous(action)


    def apply_rds_knobs(self, knobs):
        # apply knobs remotely
        db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)

        for key in knobs.keys():
            self.set_rds_param(db_conn, key, knobs[key])
        db_conn.close_db()
        return True


    def _check_apply(self, db_conn, k, v, v0, IsSession=False):
        if IsSession:
            sql = 'SHOW VARIABLES LIKE "{}";'.format(k)
            r = db_conn.fetch_results(sql)
            if r[0]['Value'] == 'ON':
                vv = 1
            elif r[0]['Value'] == 'OFF':
                vv = 0
            else:
                vv = r[0]['Value'].strip()
            if vv == v0:
                return False
            return True

        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if r[0]['Value'] == 'ON':
            vv = 1
        elif r[0]['Value'] == 'OFF':
            vv = 0
        else:
            vv = r[0]['Value'].strip()
        if vv == v0:
            return False
        return True


    def flush_status(self):
        try:
            db_conn = MysqlConnector(host=self.host,
                                 port=self.port,
                                 user=self.user,
                                 passwd=self.passwd,
                                 name=self.dbname,
                                 socket=self.sock)
            sql = 'FLUSH STATUS;'
            db_conn.execute(sql)
        except:
            pass


    def set_rds_param(self, db_conn, k, v):
        sql = 'SHOW GLOBAL VARIABLES LIKE "{}";'.format(k)
        r = db_conn.fetch_results(sql)
        if v == 'ON':
            v = 1
        elif v == 'OFF':
            v = 0
        if r[0]['Value'] == 'ON':
            v0 = 1
        elif r[0]['Value'] == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(r[0]['Value'])
            except:
                v0 = r[0]['Value'].strip()
                v0 = v0.lower()
                v = str(v).lower()

        if v0 == v:
            return True

        IsSession = False
        if str(v).isdigit():
            sql = "SET GLOBAL {}={}".format(k, v)
        else:
            sql = "SET GLOBAL {}='{}'".format(k, v)
        try:
            db_conn.execute(sql)
        except:
            logger.info("Failed: execute {}".format(sql))
            IsSession = True
            if str(v).isdigit():
                sql = "SET {}={}".format(k, v)
            else:
                sql = "SET {}='{}'".format(k, v)
            db_conn.execute(sql)
        while not self._check_apply(db_conn, k, v, v0, IsSession):
            print ( k, v, v0)
            time.sleep(1)
        return True


    def get_external_metrics(self, filename=''):
        """Get the external metrics including tps and rt"""
        result = ''
        if self.workload['name'] == 'oltpbench':
            result = parse_oltpbench('results/{}.res'.format(filename))
        elif self.workload['name'] == 'job':
            select_file = 'selectedList.txt'
            result = parse_job(filename, select_file)

        return result


    def get_benchmark_cmd(self):
        filename = self.log_path + '/{}.log'.format(self.log_file)
        dirname, _ = os.path.split(os.path.abspath(__file__))

        if self.workload['name'] == 'oltpbench':
            filename = filename.split('/')[-1].split('.')[0]
            cmd = self.workload['cmd'].format(dirname + '/cli/run_oltpbench.sh',
                                              self.dbname,
                                              self.oltpbench_config_xml,
                                              filename)
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd
        elif self.workload['name'] == 'job':
            cmd = self.workload['cmd'].format(dirname + '/cli/run_job.sh',
                                              'selectedList.txt',
                                              dirname + '/job_query/queries-mysql-new',
                                              filename,
                                              self.sock
                                              )
            cmd = "sudo cgexec -g cpuset,memory:client " + cmd

        logger.info('[DBG]. {}'.format(cmd))
        return cmd, filename


    def get_states(self):
        cmd, filename = self.get_benchmark_cmd()
        print("[{}] benchmark start!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=TIMEOUT)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                print("[{}] benchmark finished!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        except subprocess.TimeoutExpired:
            print("[{}] benchmark timeout!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        mysqladmin = os.path.join(self.mysqld) + '/mysqladmin'
        clear_cmd = str(mysqladmin) + """ processlist -uroot -S""" + str(self.sock) + """ | awk '$2 ~ /^[0-9]/ {print "KILL "$2";"}' | mysql -uroot -S""" + str(self.sock)
        subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        print("[{}] clear processlist".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if not self.connect_sucess:
            logger.info("connection failed")
            return None

        external_metrics = self.get_external_metrics(filename)

        if self.rule_based:
            self.rule.suggest(str(self.log_file)+'_2')

        return external_metrics, None, None


    def step_GP(self, knobs,  no_change=False):
        self.step_count = self.step_count + 1
        logger.info("[step {}] generate knobs: {}\n".format(self.step_count, knobs))
        collect_resource = True

        self.apply_rds_knobs(knobs)
        timestamp = int(time.time())
        self.log_file = timestamp

        if not no_change and self.rule_based:
            self.rule.suggest(self.log_file)
            knobs = self.rule.check_safe(knobs)
            self.apply_rds_knobs(knobs)

        s = self.get_states(collect_resource)

        external_metrics, internal_metrics, resource = s

        return external_metrics, internal_metrics, resource, knobs



    def terminate(self):
        return False


    def sample_workload(self, weight=[]):
        if weight == []:
            if self.fix_context:
                with open(self.context_log, 'r') as f:
                    try:
                        line = f.readlines()[self.workload_change_count]
                        weight = eval(line.strip())
                        weight.append(0)
                    except:
                        self.fix_context = False

        parser = et.parse(self.oltpbench_config_xml)
        root = parser.getroot()
        weight_text = root.findall("works")[0].findall('work')[0].findall('weights')[0].text
        numWeight = len(weight_text.split(','))
        if not len(weight) == numWeight:
            print ("Generate random workload:")
            weight = np.random.rand(numWeight,)
            weight = weight / np.sum(weight) * 100
            weight = list(weight)
        weight_str = [str(i) for i in weight]
        weight_str = ','.join(weight_str)
        root.findall("works")[0].findall('work')[0].findall('weights')[0].text = weight_str
        print(weight)
        if self.dynamic_rate:
            with open(self.rate_log, 'r') as f:
                rate = eval(f.readlines()[0])[self.workload_change_count]
                root.findall("works")[0].findall('work')[0].findall('rate')[0].text = rate

        parser.write(self.oltpbench_config_xml)

        if not self.fix_context:
            with open(self.context_log, 'a') as f:
                f.write(weight)

        self.workload_change_count = self.workload_change_count + 1

        return weight


    def set_general_log(self, value=1):
        knob = {}
        knob['general_log'] = value
        knob['max_connections'] = 1000
        try:
            self.apply_rds_knobs(knob)
        except:
            logger.info("apply rds knobs failed, re-started")
            self.apply_knobs(knob)


    def collect_plan(self):
        cmd = "bash ../Workload2Vec/gen_plan.sh {} {} {}".format(self.log_file, self.dbname, self.sock)
        logger.info(cmd)
        cmd_result = os.popen(cmd)
        cmd_result = cmd_result.read()
        try:
            plan_vec = cmd_result.splitlines()[-3]
            plan_vec = eval(plan_vec)
        except:
            plan_vec = [0] * 6

        return plan_vec


    def collect_LSTM(self):
        if self.workload['name'] == 'job':
            cmd = 'cd ../Workload2Vec/LSTM; python sql_prepare.py test'
        else:
            cmd = "bash ../Workload2Vec/run_all_LSTM.sh {}".format(self.log_file)
        logger.info(cmd)
        subprocess.call(cmd, shell=True)
        workload_embedding = self.get_embedding().mean(axis=0)
        return workload_embedding


    def cp_general_log(self):
        cmd = "mv ../Workload2Vec/logs/general.log ../Workload2Vec/logs/general.log_{}".format(self.log_file)
        os.system(cmd)

    def get_embedding(self):
        sentences, sizes = encoder_utils.load_text_data('../Workload2Vec/LSTM/test.sql', self.wd)
        num_sents = 1000
        next_index = 0
        all_states = []
        while next_index < len(sentences):
            batch = sentences[next_index:next_index + num_sents]
            batch_sizes = sizes[next_index:next_index + num_sents]
            next_index += num_sents
            state = self.encoder.encode(self.sess, batch, batch_sizes)
            all_states.append(state)

        state = np.vstack(all_states)

        return state

    def sample_workload_job(self, weight=[]):
        if not len(weight):
            with open(self.context_log, 'r') as f:
                line = f.readlines()[self.workload_change_count]
                weight = eval(line.strip())
                self.workload_change_count = self.workload_change_count + 1

        select_file = 'selectedList.txt'
        f_select_file = open(select_file, 'w')
        f_test_file = open('../Workload2Vec/LSTM/test', 'w')
        dirname, _ = os.path.split(os.path.abspath(__file__))
        dir = os.path.join(dirname, 'job_query/queries-mysql')
        for w in weight:
            f_select_file.write(w + '\n')
            f = open(os.path.join(dir, w))
            lines = f.readlines()
            line = ' '.join(lines)
            f_test_file.write(line)

        f_select_file.close()
        f_test_file.close()

        return weight


    def collec_default(self, weight=[], get_embedding=True, get_state=False):
        self.set_general_log("OFF")
        if get_embedding:
            workload_vec =  self.collect_LSTM()
            plan_vec = self.collect_plan()
            if self.workload['name'] == 'job':
                context = gen_context(workload_vec, plan_vec, len(weight))
            else:
                context = gen_context(workload_vec, plan_vec, metrics[0])
        else:
            workload_vec, plan_vec, context= [], [], []

        self.cp_general_log()
        if os.path.exists('../Workload2Vec/logs/general.log'):
            os.system("rm ../Workload2Vec/logs/general.log")

        self.set_general_log("ON")

        if self.workload['name'] == 'job':
             weight = self.sample_workload_job(weight)
        else:
            weight = self.sample_workload(weight)

        logger.info("generate workload {}, its weight is {}, plan is {}".format(self.log_file, weight, plan_vec))


        return workload_vec,  plan_vec, context



