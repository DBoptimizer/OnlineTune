OLTPBENCH_WORKLOADS = {
    'name': 'oltpbench',
    'type': 'oltpbenchmark',
    # bash run_oltpbench.sh benchmark config_xml output_file
    'cmd': 'bash {} {} {} {}'
}


JOB_WORKLOAD = {
    'name': 'job',
    'type': 'read',
    # bash run_job.sh queries_list.txt query_dir output.log MYSQL_SOCK
    'cmd': 'bash {} {} {} {} {}'
}
