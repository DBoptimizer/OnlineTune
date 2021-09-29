# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals

import argparse
import os
import numpy as np
import math
import re
from collections import defaultdict, Counter
import pdb
from nltk.tokenize import RegexpTokenizer
"""
This script processes an input text file to produce data in binary
format to be used with the autoencoder (binary is much faster to read).
"""
keywordList = {'select', 'update', 'delete', 'insert', 'replace'}

pre_LSTM = True

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def alis(line):
    if not 'from' in line or 'insert' in line:
        return  line
    begin = line.find('from') + 4
    end = line.find('where')
    tables = line[begin:end]
    if 'select' in tables:
        return line
    tmp = tables.split(',')
    for t in tmp:
        t = t.strip()
        if len(t.split()) < 2:
            continue
        table = t.split(" ")[0]
        try:
            alis = t.split(" ")[1]
        except:
            pdb.set_trace()
        if alis == 'order':
            continue
        line = line.replace(t, table)
        tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\.|=|>|<|\S+')
        tokens = tokenizer.tokenize(line)
        tokens = [table if i == alis else i for i in tokens]
        line = ' '.join(tokens)
        line = line + '\n'

    return line



def transfer_sql(path):
    fileLen = file_len(path)
    print("We have {} lines to deal with".format(fileLen))
    tokenL, lineL = [], []
    count = 0
    das_flag = False
    f_out = path + '.sql'
    f_out = open(f_out, 'w')
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').lower()
            if count == 0 and ",log" in line:
                das_flag = True
            #pdb.set_trace()
            sqlFlag = True if any(i in line for i in keywordList) else False
            count = count + 1
            if count%100000 == 0:
                print('finish {}%'.format(float(count / fileLen * 100)))
            if not sqlFlag:
                if not pre_LSTM:
                    f_out.write(line.strip() + '\n')
                continue
            if "mysql-connector-java" in line :#or 'tx_read_only' in line:
                if not pre_LSTM:
                    f_out.write(line.strip() + '\n')
                continue
            if pre_LSTM and 'tx_read_only' in line:
                continue

            if not das_flag:
                begin_ind = line.find('\t', len(line.split('\t')[0]) + 1)
                begin_ind = 0 if begin_ind == -1 else begin_ind
                line = line[begin_ind:].strip()
                line = line.strip().strip(';').strip(')')
            else:
                begin_ind = line.find(',') + 1
                line = line[begin_ind:].strip()
                # if line[0] == '"' and line[-1] == '"':
                line = line.strip('"')

            line = alis(line)
            line = line.replace('(', ' ')
            line = line.replace(')', ' ')
            line_o = line
            if not ('ycsb_key' in line or 'insert into usertable' in line):
                p2 = re.compile(r'[\'](.*?)[\']', re.S)
            else:
                p2 = re.compile(r'[\'](.*)[\']', re.S)
            value = re.findall(p2, line)
            for v in value:
                line = line.replace('\''+v+'\'', 'VALUE_TMP')
            tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\.|=|>|<|,|\S+')
            tokens = tokenizer.tokenize(line)
            tokens = [i.strip() for i in tokens]
            tokens = ['<DIGIT>' if is_number(i.strip(',')) else i for i in tokens]
            tokens = ['<VALUE>' if ( len(i) >=1 and i[0]=="\'" and i[-1]=="\'") else i for i in tokens]
            line = ' '.join(tokens)
            line = line.replace('VALUE_TMP', '<VALUE>')
            if "'" in line:
                line = line_o
                p2 = re.compile(r'[\'](.*)[\']', re.S)
                value = re.findall(p2, line)
                for v in value:
                    line = line.replace('\'' + v + '\'', 'VALUE_TMP')

                tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\.|=|>|<|,|\S+')
                tokens = tokenizer.tokenize(line)
                tokens = [i.strip() for i in tokens]
                tokens = ['<DIGIT>' if is_number(i.strip(',')) else i for i in tokens]
                tokens = ['<VALUE>' if (len(i) >= 1 and i[0] == "\'" and i[-1] == "\'") else i for i in tokens]
                line = ' '.join(tokens)
                line = line.replace('VALUE_TMP', '<VALUE>')

            #line = line.replace("'<VALUE>',",  '<VALUE>')
            line = line + '\n'
            #tokenL.append(tokens)
            lineL.append(line)
            f_out.write(line)
            if "field8" in line:
                pdb.set_trace()

    f_out.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file previously tokenized '
                                      '(by whitespace) and preprocessed')

    args = parser.parse_args()


    sql = transfer_sql(args.input)

