# OnlineTune
 Online configuration tuning system for DBMS with safety consideration

## Installation

```
pip install -r requirements.txt
pip install setup.py
```



## Usgae

To run OnlineTune locally:
```
cd scripts
python optimize.py --knobs_config=knobs.json --benchmark=oltpbench_tpcc --data=history.result  --y_variable=tps
```
