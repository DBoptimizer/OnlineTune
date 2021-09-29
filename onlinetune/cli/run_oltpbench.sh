#!/usr/bin/env bash
OLTPBENCH_BIN='/data2/ruike/oltpbench-master/oltpbenchmark'

$OLTPBENCH_BIN -b ${1} -c ${2} --execute=true -s 1 -o ${3}
