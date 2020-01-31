import sys
import os
import subprocess
import time
import datetime
import copy
import re
import pprint

pp = pprint.PrettyPrinter(indent=2)

# Syntax:
# python tracker/experiments.py path/to/expdir/ [single/cluster]


expd = sys.argv[1]
if not expd.endswith("/"):
    expd += "/"

# optional arguments
skip_combinations = []
copyfiles = {}

workers_per_server = [0]
print("Running experiments defined in " + expd)
execfile(expd + "settings.py")

# option to run only single-machine or cluster experiments
focus = ""
if len(sys.argv) > 2:
    focus = sys.argv[2]
    print("Focus: " + focus)

    if focus == "single":
        launcher = "python tracker/dmlc_local.py"
        print("Override launcher for single machine:\n"+launcher)


if 'compile' in locals():
    for c in compile:
        command = c
        subprocess.call(command, shell=True)


explist = []
expargs = []


def matches(expargs, check_arg, check_val):
    for arg, val in expargs:
        if arg == check_arg and val == check_val:
            return True
    return False


def addarg(arguments, expargs, command):
    if not arguments:
        # check whether this combination should be skipped
        skip = False
        if skip_combinations:
            for skip_combination in skip_combinations:
                all_match = True
                for check_arg, check_val in skip_combination.iteritems():
                    all_match = all_match and matches(expargs, check_arg, check_val)
                if all_match:
                    skip = True
                    print("Condition")
                    print(skip_combination)
                    print("causes skip of experiment run:")
                    print(expargs)
                    print("")
        if not skip:
            explist.append((copy.deepcopy(expargs), copy.deepcopy(command)))
    else:
        arg, vals = arguments.pop()
        if not isinstance(vals, list):
            vals = [vals]
        for v in vals:
            command.append("--" + str(arg) + " " + str(v))
            expargs.append((arg, v))
            addarg(arguments, expargs, command)
            expargs.pop()
            command.pop()
	arguments.append((arg,vals))

for rep in xrange(reps):
    expargs.append(("rep", rep))
    for e in executable:
        expargs.append(("executable", e))
        for s in servers:
            if focus == "single" and s > 1:
                print("Skip cluster experiment runs")
                continue
            if focus == "cluster" and s == 1:
                print("Skip single-machine experiment runs")
                continue
            expargs.append(("servers", s))

	    for wps in workers_per_server:
		expargs.append(("workers_per_server", wps))
		workers = wps * s

		# create experiment list recursively
		command = [launcher + " -s " + str(s) + " -n " + str(workers) + " " + e]
		arglist = arguments.items()
		addarg(arglist, expargs, command)

		expargs.pop()
            expargs.pop()
        expargs.pop()
    expargs.pop()


def findmatches(filename, regex):
    matches = []
    with open(filename, "r") as f:
        for line in f:
            if re.match(regex, line):
                m = re.search(regex, line)
                matches.append((line, m.groups()))

    print(str(len(matches)) + " matches: \t" + regex)
    return matches

def lastmatch(filename, regex):
    matches = findmatches(filename, regex)
    if len(matches) == 0:
	return float('nan')
    return matches[-1][1][0]


id = 0
headers = [k for k,v in explist[0][0]]
headers = ["id"] + headers
headers.append("returncode")
for measure, regex in measures.items():
    headers.append(measure)

for analysis_key, analysis in analyses.items():
    for measure, regex in analysis["measures"].items():
	headers.append(analysis_key + measure)

print(headers)
measurements_file = expd + "measurements" + ("." if len(focus) > 0 else "") + focus + ".csv"
with open(measurements_file, "w+") as f:
    f.write("\t".join(headers) + "\n")

if not os.path.isdir(expd + "logs"):
    os.mkdir(expd + "logs")

# note code version
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
with open(expd + "git_head.txt", "w+") as f:
    f.write(head)
diff = subprocess.check_output(['git', 'diff'])
with open(expd + "git_diff.txt", "w+") as f:
    f.write(diff)

# run experiments
for exp in explist:
    arguments, command = exp
    logfile = expd + "logs/" + str(id) + ".log"
    command = " ".join(command) + " &> " + logfile

    print("\n----------------------------------------------------\n")
    print("ID " + str(id) + " / " + str(len(explist)))
    print(str(datetime.datetime.now()))
    for arg, val in arguments:
        print(str(arg) + ": " + str(val))
    print ("")
    print(command + "\n")
    returncode = subprocess.call(command, shell=True)

    data = [str(v) for k,v in arguments]
    data = [str(id)] + data
    data.append(str(returncode))

    for measure, regex in measures.items():
        value = lastmatch(logfile, regex)
        data.append(str(value))

    for copyfile, copytarget in copyfiles.items():
	copycmd = "cp " + copyfile + " " + expd + "logs/" + str(id)+"_"+copytarget
	print("Copy: " + copycmd)
	subprocess.call(copycmd, shell=True)

    for analysis_key, analysis in analyses.items():
	analysis_file = expd + "logs/" + str(id) + ".analysis." + analysis_key + ".log"
        analysis_cmd = analysis["command"] + " &> " + analysis_file
	print("\nRun external analysis: " + analysis_cmd)
        subprocess.call(analysis_cmd, shell=True)
	for measure, regex in analysis["measures"].items():
	    value = lastmatch(analysis_file, regex)
	    data.append(str(value))

    with open(measurements_file, "a+") as f:
        f.write("\t".join(data) + "\n")

    id = id + 1


