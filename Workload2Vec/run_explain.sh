#!/usr/bin/env bash
# run_job.sh  selectedList.txt  queries_dir   output	MYSQL_SOCK
MYSQL_SOCK=${1}
PASSWD=${2}
DBNAME=${3}
QUERY_FILE=${4}
OUTPUT_FILE=${5}
SECONDS=0 ;



for file in `ls $QUERY_FILE `
do
  {
file=${QUERY_FILE}'/'${file}
mysql -sN -S$MYSQL_SOCK -uroot -p$PASSWD $DBNAME < $file > ${file}'_output' 2>/dev/null
} &
done


echo $SECONDS  "seconds passed"

cat ${QUERY_FILE}"/"*'output' > $OUTPUT_FILE