import pandas as pd
import statistics
import time


def parse_job(file_path, select_file):
    with open(file_path) as f:
        lines = f.readlines()

    with open(select_file) as f:
        lines_select = f.readlines()
    num_sql = len(lines_select)

    latL = []
    for line in lines[1:]:
        if line.strip() == '':
            continue
        tmp = line.split('\t')[-1].strip()
        latL.append(float(tmp)/1000)

    sql_dealed = len(latL)
    tpm = len(latL) / 3
    for i in range(0, num_sql - sql_dealed):
        latL.append(180)
    lat = np.percentile(latL, 90)

    lat_var = statistics.variance(latL)

    return [tpm, lat, tpm, -1, lat_var, -1]


def parse_oltpbench(file_path):
    # file_path = *.summary
    flag = True
    count = 0
    while flag:
        try:
            if count >90:
                print('num_samples is zero!')
                return [-1, -1, -1, -1, -1, -1]
            df = pd.read_csv(file_path, delimiter = ",")
            flag = False
        except:
            print ("sleep {}".format(count*2))
            time.sleep(2)
            count = count + 1

    tps = df[' throughput(req/sec)'].mean()
    tps_var = df[' throughput(req/sec)'].var()
    lat = df[' 95th_lat(ms)'].mean()
    lat_var = df[' 95th_lat(ms)'].var()

    return [tps, lat, tps, tps_var, lat_var, tps_var]





