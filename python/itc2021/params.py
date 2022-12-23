from typing import List
import sys
import time
import random

class Parameters:

    def __init__(self, args: List[str]):
        # default parameter values
        self.approach = "fixopt"
        self.emphasis = 1
        self.instance_file = None
        self.initsoln_file = None
        self.output = "output.xml"
        self.threads = -1
        self.timelimit = 30
        self.fixopt_timelimit = 200
        self.fixopt_n = 10
        self.fixopt_neighborhood = "s"
        self.local_branching = 0
        self.local_branching_k = 10
        self.seed = None
        self.verbose = 0
        self.version = 'v3'

        self.starttime = time.time()

        if not self.read_args(args):
            self.print_usage()
            sys.exit(0)

    def print_params(self):
        print("Parameter values:")
        print("     approach.:", self.approach)
        print("     emphasis.:", self.emphasis)
        print("     fixopt_n:", self.fixopt_n)
        print("     fixopt_timelimit:", self.fixopt_timelimit)
        print("     output...:", self.output)
        print("     threads..:", self.threads)
        print("     timelimit:", self.timelimit)
        print("     version:", self.version)
        print()

    def print_usage(self):
        print(f"Usage: python3 main.py <input> [options]")
        print(f"    <input> : Path of the problem input file.")
        print(f"")
        print(f"Options:")
        print(f"  -approach <approach>          : selected approach/formulation to execute; possible values are")
        print(f"                                  {{mathprog, fixopt, sls}} (default: {self.approach})")
        print(f"  -emphasis <emphasis>          : search emphasis (default: {self.emphasis}).")
        print(f"  -initsoln_file <filepath>     : folder to load initial solution from (default: {self.initsoln_file}).")
        print(f"  -output <filepath>            : output folder to save solutions and additional info (default: {self.output}).")
        print(f"  -threads <threads>            : maximum number of threads (default: {self.threads}).")
        print(f"  -timelimit <timelimit>        : runtime limit (default: {self.timelimit}).")
        print(f"  -fixopt_timelimit <timelimit> : time limit for each fixopt iteration (default: {self.fixopt_timelimit}).")
        print(f"  -fixopt_n <n>                 : number of teams to be optimized at each iteration of fixopt (default: {self.fixopt_n}).")
        print(f"  -fixopt_neighborhood <neigh>  : neighborhood of fixopt algorithm (t; s; m; all) (default: {self.fixopt_neighborhood}).")
        print(f"  -local_branching <n>          : activate or not local branching {{0, 1}} (default: {self.local_branching}).")
        print(f"  -local_branching_k <n>        : parameter k for local branching (default: {self.local_branching_k}).")
        print(f"  -seed <seed>                  : random seed (default: {self.seed}).")
        print(f"  -verbose <verbose>            : print MIP solver logs (0/1) (default: {self.verbose}).")
        print(f"  -version <version>            : model/code version (default: {self.version}).")
        print(f"")

    def read_args(self, args) -> bool:
        print("Arguments: %s\n" % " ".join(args))

        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "-approach":
                self.approach = sys.argv[i+1]
                print("Approach value set to '%s'" % self.approach)
                i += 2
            elif sys.argv[i] == "-emphasis":
                self.emphasis = int(sys.argv[i+1])
                print("Emphasis set to %d" % self.emphasis)
                i += 2
            elif sys.argv[i] == "-initsoln_file":
                self.initsoln_file = sys.argv[i + 1]
                print("Initial soln folder set to %s" % self.initsoln_file)
                i += 2
            elif sys.argv[i] == "-output":
                self.output = sys.argv[i+1]
                print("Output folder set to %s" % self.output)
                i += 2
            elif sys.argv[i] == "-threads":
                self.threads = int(sys.argv[i+1])
                print("Number of threads set to %d" % self.threads)
                i += 2
            elif sys.argv[i] == "-timelimit":
                self.timelimit = int(sys.argv[i+1])
                print("Runtime limit set to %d seconds" % self.timelimit)
                i += 2
            elif sys.argv[i] == "-fixopt_timelimit":
                self.fixopt_timelimit = int(sys.argv[i + 1])
                print("Fixopt time limit set to %d seconds" % self.fixopt_timelimit)
                i += 2
            elif sys.argv[i] == "-fixopt_n":
                self.fixopt_n = int(sys.argv[i + 1])
                print("Fixopt n parameter set to %d" % self.fixopt_n)
                i += 2
            elif sys.argv[i] == "-fixopt_neighborhood":
                self.fixopt_neighborhood = sys.argv[i + 1]
                print("Fixopt negihborhood set to %s" % self.fixopt_neighborhood)
                i += 2
            elif sys.argv[i] == "-local_branching":
                self.local_branching = int(sys.argv[i + 1])
                print("Local branching set to %d" % self.local_branching)
                i += 2
            elif sys.argv[i] == "-local_branching_k":
                self.local_branching_k = int(sys.argv[i + 1])
                print("Local branching k parameter set to %d" % self.local_branching_k)
                i += 2
            elif sys.argv[i] == "-seed":
                self.seed = int(sys.argv[i + 1])
                random.seed = self.seed
                print("Random seed set to %d" % self.seed)
                i += 2
            elif sys.argv[i] == "-verbose":
                self.verbose = int(sys.argv[i + 1])
                print("Verbose logs set to %d" % self.verbose)
                i += 2
            elif sys.argv[i] == "-version":
                self.version = sys.argv[i + 1]
                print("Version set to %s" % self.version)
                i += 2
            elif not self.instance_file:
                self.instance_file = sys.argv[i]
                print("Reading input file %s" % self.instance_file)
                i += 1
            else:
                print("\nERROR: Unrecognized argument '%s'\n" %
                      sys.argv[i], file=sys.stderr)
                return False

        if not self.instance_file:
            print("ERROR: No instance file provided!\n", file=sys.stderr)
            return False

        print()
        return True
