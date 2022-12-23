import glob, os, sys


def main(argv):
    if len(argv) == 1:
        usage()

    folder = argv[1]
    files = glob.glob(f"{folder}/*.lp")
    for file in files:
        print(f"Processing {file}...")

        # creating gurobi script
        script = open(f"{file}.gurobi", "w")
        print(f"from gurobipy import *", file=script)
        print(f"model = read('{file}')", file=script)
        print(f"model.relax().optimize()", file=script)
        print(f"exit()", file=script)
        script.close()

        # executing...
        os.system(f"gurobi.sh {file}.gurobi > {file}.log 2>&1")
        os.remove(f"{file}.gurobi")

        # reading LP relaxation value
        lb, iters, runtime = -float("inf"), -1, -1
        log = open(f"{file}.log", "r")
        for line in log.readlines():
            if line.startswith('Optimal objective'):
                lb = float(line.split()[-1])
            elif line.startswith('Solved in'):
                iters = int(line.split()[2])
                runtime = float(line.split()[5])
        log.close()

        # creating header for output file
        if not os.path.exists(f"lp_log.txt"):
            out = open(f"lp_log.txt", "w")
            print(f"{'Instance/file name':30s} {'Iters':-10s} {'Runtime:-10s'} {'LP Relax.:-10s'}", file=out)
            out.close()

        # printing to log file
        out = open(f"lp_log.txt", "a")
        print(f"{file:30s} {iters:10d} {runtime:10.1f} {lb:10.3f}", file=out)
        print(f"{file:30s} {iters:10d} {runtime:10.1f} {lb:10.3f}")
        out.close()


def usage():
    print(f"Usage: python3 main.py <solve_relax>")
    print(f"    <folder> : Path with LP files.")
    print(f"")
    exit(-1)


if __name__ == "__main__":
    main(sys.argv)
