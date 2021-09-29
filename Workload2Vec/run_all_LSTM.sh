LOG_FILE=${1}
General_Log='../logs/general.log'
Dir="../Workload2Vec/LSTM"

export TF_CPP_MIN_LOG_LEVEL=3
cd $Dir

a=`wc -l  $General_Log`
ARR=($a)
n=${ARR[0]}
let n=$n/100
shuf $General_Log -n $n -o test
python sql_prepare.py test