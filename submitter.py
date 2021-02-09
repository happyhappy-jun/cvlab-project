#!python3
import subprocess, codecs, time
from time import sleep
from pygtail import Pygtail
import os, sys

def follow(thefile):
    '''generator function that yields new lines in a file
    '''
    # seek the end of the file
    thefile.seek(0, os.SEEK_END)
    
    # start infinite loop
    while True:
        # read last line of file
        line = thefile.readline()
        # sleep if file hasn't been updated
        if not line:
            time.sleep(0.1)
            continue

        yield line

def split_space(x):
    if not isinstance(x, str):
        x = x.decode('utf-8')
    return [item.strip() for item in x.strip().split(" ") if item != ""]

subp = subprocess.Popen("squeue | grep junyoon", shell=True, stdout=subprocess.PIPE)
subprocess_return = subp.stdout.read().strip()
if len(subprocess_return) != 0:
    jobs = []
    for x in subprocess_return.decode("utf-8").split("\n"):
        print(x)
        line = split_space(x)
        job = {"jobid": line[0], "gpu": line[1], "jobname": line[2], "user": line[3], "status": line[4], "node": line[6]}
        jobs.append(job)
    print(jobs)
    for j in jobs:
        if j["status"] == "R":
            os.system(f"scancel {j['jobid']}")
        

#submit job
subp = subprocess.Popen("sbatch experiment.sh", shell=True, stdout=subprocess.PIPE)
subprocess_return = subp.stdout.read().strip()
print(subprocess_return)
submitted_id = split_space(subprocess_return)[-1]

sleep(2)
# show log

logfile = open(f"{submitted_id}.out","r")
loglines = follow(logfile)