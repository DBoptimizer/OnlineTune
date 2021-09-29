import argparse
from onlinetune.dbenv import MySQLEnv
from onlinetune.workload import SYSBENCH_WORKLOAD, TPCC_WORKLOAD, JOB_WORKLOAD
from onlinetune.workload import OLTPBENCH_WORKLOADS, WORKLOAD_ZOO_WORKLOADS
from onlinetune.tuner_safe import SafeMySQLTuner
from onlinetune.knobs import logger
from onlinetune.utils.helper import check_env_setting
from onlinetune.utils.autotune_exceptions import AutotuneError


HOST='localhost'
THREADS=64
PASSWD=''
PORT=3306
USER=''
LOG_PATH="./log"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='tpcc', help='[sysbench, tpcc, workload_zoo, \
                        oltpbench_wikipedia, oltpbench_syntheticresourcestresser, oltpbench_twitter, oltpbench_tatp, \
                        oltpbench_auctionmark, oltpbench_seats, oltpbench_ycsb, oltpbench_jpab, \
                        oltpbench_ch-benchmark, oltpbench_voter, oltpbench_slbench, oltpbench_smallbank, oltpbench_linkbench]')
    parser.add_argument('--knobs_config', type=str, default='', help='knobs configuration file in json format')
     parser.add_argument('--data', type=str, default='', help='historical data')
    parser.add_argument('--oltpbench_config_xml', type=str, default='', help='config_xml for OLTPBench')
    parser.add_argument('--pid', type=int, default=9999, help='mysql pid')
    parser.add_argument('--knobs_num', type=int, default=-1, help='knobs num')
    parser.add_argument('--y_variable', type=str, default='tps', help='[tps, lat]')


    opt = parser.parse_args()

    if opt.knobs_config == '':
        err_msg = 'You must specify the knobs_config file for tuning: --knobs_config=config.json'
        logger.error(err_msg)
        raise AutotuneError(err_msg)




    # Check env
    check_env_setting(opt.benchmark, True)

    wl = None
    dbname = opt.dbname
    if opt.benchmark == 'tpcc':
        wl = dict(TPCC_WORKLOAD)
    elif opt.benchmark == 'sysbench':
        wl = dict(SYSBENCH_WORKLOAD)
        wl['type'] = opt.workload_type
    elif opt.benchmark.startswith('oltpbench_'):
        wl = dict(OLTPBENCH_WORKLOADS)
        dbname = opt.benchmark[10:]  # strip oltpbench_
        logger.info('use database name {} by default'.format(dbname))
    elif opt.benchmark == 'workload_zoo':
        wl = dict(WORKLOAD_ZOO_WORKLOADS)
    elif opt.benchmark == 'job':
        wl = dict(JOB_WORKLOAD)



    env = MySQLEnv(workload=wl,
                    knobs_config=opt.knobs_config,
                    num_metrics=65,
                    log_path=LOG_PATH,
                    threads=THREADS,
                    host=HOST,
                    port=PORT,
                    user=USER,
                    passwd=PASSWD,
                    dbname=dbname,
                    rds_mode=True,
                    oltpbench_config_xml=opt.oltpbench_config_xml,
                    pid=opt.pid,
                    knob_num=opt.knobs_num)

    env.context_log = '../setting/change_6_var_0.1.w'
    logger.info('env initialized with the following options: {}'.format(opt))


    tuner = SafeMySQLTuner(env=env, data=opt.data, y_variable=opt.y_variable)
    tuner.tune()

