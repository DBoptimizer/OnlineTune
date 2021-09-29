#this file is used for extracting the queries from general file

import ipdb
import argparse
import re
import os
import time

THREAD = 64

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

parser = argparse.ArgumentParser(description='input the inputfileName')
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

keywordList = {'select', 'update', 'delete', 'insert', 'replace'}


time_begin = time.time()
fileLen = file_len(args.input)
print ("We have {} lines to deal with".format(fileLen))

fileL = []
for i in range(0, THREAD):
    f = open(args.output+'/tmp'+str(i), 'w')
    fileL.append(f)


with open(args.input, 'r') as fin:
    count = 0
    query_count = 0
    for line in fin:
        newline = line.lower()
        sqlFlag = True if any(i in newline for i in keywordList) else False
        count = count + 1
        if not sqlFlag:
            continue
        if "mysql-connector-java"  in line:
            continue
        begin_ind = newline.find('\t', len(newline.split('\t')[0])+1)
        newline = line[begin_ind:].strip()
        newline = "explain " + newline + ';'
        writeFile = fileL[query_count%THREAD]
        writeFile.write(newline + '\n')
        query_count = query_count + 1
        if count%100000 == 0:
            print ('finish {}%'.format(float(count/fileLen*100)))

for f in fileL:
    f.close()

print ("{} seconds passed".format(time.time() - time_begin))
print ("You extract {} queries".format(query_count))