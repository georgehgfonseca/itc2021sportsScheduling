import sys
from instance import Instance
from params import Parameters
from solution import Solution
import time


def main(args):
    if len(args) == 1:
        args = args + ['../../data/TestInstanceDemo.xml']

    # Reading parameters and instance
    params = Parameters(sys.argv)
    inst = Instance(args[1])

    # Printing basic statistics
    print(f"Number of x vars: {len(inst.teams) * (len(inst.teams) - 1) * len(inst.slots)}")
    print("Loading initial solution...")

    # Reading input solution (if any is specified)
    if params.initsoln_file is not None:
        try:  # Load soln from file
            soln, inf, obj = Solution.read_xml_soln(params.initsoln_file)
        except IOError as e:  # Build initial soln
            print(f"File {params.initsoln_file} could not be found! Building solution instead")
            soln, inf, obj = Solution.build_init_soln(inst)

    # If no input solution is specified, building a potentially infeasible one using a simple constructive approach
    else:
        soln, inf, obj = Solution.build_init_soln(inst)

    # Importing model
    if params.version == 'v1':
        from mipModelV1 import STTModel
    elif params.version == 'v2':
        from mipModelV2 import STTModel
    elif params.version == 'v3':
        from mipModelV3 import STTModel
    elif params.version == 'v4':
        from mipModelV4 import STTModel
    else:
        print(f"Code base {params.version} is unknown. Aborting...")
        exit(-1)

    # Creating empty model
    mip1 = None

    # Running phase 1 (optimizing hard constraints) if infeasibility is detected
    if inf != 0:
        print("Phase 1:")
        mip1 = STTModel(inst, params, 1, soln)

    # Running phase 2 (optimizing soft constraints)
    if time.time() - params.starttime < params.timelimit:
        print("Phase 2:")
        if mip1 is not None:
            mip2 = STTModel(inst, params, 2, mip1.soln)
        else:
            mip2 = STTModel(inst, params, 2, soln)

    # soln = Solution(mip.soln, mip.ub)
    # soln.write(params.output)


if __name__ == "__main__":
    main(sys.argv)
