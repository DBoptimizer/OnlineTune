MYSQL_SOCK=${3}
GENERAL_LOG='logs/general.log'
PASSWD=123456
DBNAME=${2}
OUTPUT_FILE='plan/plan.log'
DIR='../Workload2Vec'
LOG_FILE=${1}

SECONDS=0 ;
cd $DIR
if  [ ! -d "queryDir" ];
then
 mkdir "queryDir"
fi
python sqlExtracting.py  --input=$GENERAL_LOG --output="queryDir"
mysql -sN -S$MYSQL_SOCK -uroot -p$PASSWD $DBNAME -e"set global general_log=0;"
TMP='_'
#mv $GENERAL_LOG "$GENERAL_LOG$TMP$LOG_FILE"
bash run_explain.sh $MYSQL_SOCK $PASSWD $DBNAME "queryDir"  "$OUTPUT_FILE$TMP$LOG_FILE"
python plan2vec.py  --input="$OUTPUT_FILE$TMP$LOG_FILE"
rm -rf "queryDir"
echo $LOG_FILE
echo $SECONDS  "seconds passed"