import mip
from instance import Instance
from params import Parameters
import numpy as np
from mip import Var, Model, BINARY, INTEGER, CONTINUOUS, xsum, OptimizationStatus
from itertools import permutations
import random
import time
from solution import Solution

class STTModel:
    __slots__ = ['inst', 'params', 'model', 'x', 'bH', 'bA', 's', 'soln', 'ub']

    def __init__(self, inst: Instance, params: Parameters, soln=None):
        self.inst = inst
        self.params = params
        self.soln = soln

        # model-related variables
        self.model = None
        self.x = self.bH = self.bA = self.s = None
        self.ub = float("inf")

        # creating and solving model
        self.create_model()
        if self.soln != None:
            self.load_soln()
        if self.params.approach == "fixopt":
            self.fixopt()
        elif self.params.approach == "mathprog":
            self.mathprog()

    def mathprog(self):
        self.set_solver_options()
        self.model.write(self.params.instance_file + ".lp")
        self.model.optimize(max_seconds=self.params.timelimit)
        self.print_vars()
        if self.model.num_solutions > 0 and self.model.objective_value < self.ub:
            self.ub = self.model.objective_value
            self.convert_soln()
            soln = Solution(self.soln, self.ub)
            soln.write(self.params.output)

    def fixopt(self):
        T, S = self.inst.teams, self.inst.slots
        it = 0
        while(time.time() - self.params.starttime < self.params.timelimit):
            it += 1
            #select vars to be unfixed
            Tline = []
            Taux = [x for x in range(len(T))]
            if self.params.fixopt_n >= len(T) - 1:
                Tline = Taux
            else:
                while len(Tline) < self.params.fixopt_n:
                    t = random.choice(Taux)
                    Taux.remove(t)
                    Tline.append(t)
            print(Tline)

            #fix all x vars except the selected ones (teams in Tline)
            for i in range(len(self.inst.teams)):
                for j in range(len(self.inst.teams)):
                    if i != j:
                        for k in range(len(self.inst.slots)):
                            if i in Tline or j in Tline:
                                self.x[i, j, k].lb = 0
                                self.x[i, j, k].ub = 1
                            else:
                                if (self.x[i,j,k].x != None and self.x[i,j,k].x >= 0.999) or ((i, j, k) in self.soln and it == 1):
                                    self.x[i,j,k].lb = self.x[i,j,k].ub = 1
                                else:
                                    self.x[i,j,k].lb = self.x[i,j,k].ub = 0

            #solve model
            #self.model.write('model_stt.lp')
            self.set_solver_options()
            max_time = min(self.params.fixopt_timelimit, self.params.timelimit - (time.time() - self.params.starttime))
            self.model.optimize(max_seconds=max_time)
            # self.print_vars()

            #auto-adaption of parameter n
            if self.model.status == OptimizationStatus.OPTIMAL and self.model.objective_value < self.ub: #Optimal solution found (new best)
                pass
            elif self.model.status == OptimizationStatus.OPTIMAL: #Optimal solution found (but not better than previous)
                self.params.fixopt_n += 1
            elif self.model.status != OptimizationStatus.OPTIMAL: #Optimal solution has not been found (possible due to timelimit)
                self.params.fixopt_n -= 1

            #write solution if it is a new best
            if self.model.num_solutions > 0 and self.model.objective_value < self.ub:
                print(f"New best of cost {self.model.objective_value} found within {time.time() - self.params.starttime} secs!")
                self.ub = self.model.objective_value
                self.convert_soln()
                soln = Solution(self.soln, self.ub)
                soln.write(self.params.output)
            if self.model.status == OptimizationStatus.OPTIMAL and self.params.fixopt_n >= len(T) - 1:
                print("Optimal solution found!")
                return

    def load_soln(self):
        mipStart = []
        for i in range(len(self.inst.teams)):
            for j in range(len(self.inst.teams)):
                if i != j:
                    for k in range(len(self.inst.slots)):
                        if (i, j, k) in self.soln:
                            mipStart.append((self.x[i, j, k], 1))
                        else:
                            mipStart.append((self.x[i, j, k], 0))
        self.model.start = mipStart

        # for s in self.soln:
        #     i = s[0]
        #     j = s[1]
        #     k = s[2]
        #     mipStart.append((self.x[i, j, k], 1))
        #     #Fix initial solution (for debug)
        #     #self.x[i,j,k].lb = self.x[i,j,k].ub = 1
        # self.model.start = mipStart

    def convert_soln(self):
        self.soln = []
        for i in range(len(self.inst.teams)):
            for j in range(len(self.inst.teams)):
                if i != j:
                    for k in range(len(self.inst.slots)):
                        if self.x[i, j, k].x >= 0.00001:
                            self.soln.append((i, j, k))

    def create_model(self):
        self.model = model = Model("STT")

        # aliases and shortcuts to make code easier to read
        inst, T, S, number_rr, compactness, game_mode, C = self.inst, self.inst.teams, self.inst.slots, self.inst.number_rr, self.inst.compactness, self.inst.game_mode, self.inst.constrs

        # # creating lower and upper bounds for variables *u*
        # self.lbs = lbs = np.array([1] + [2 for j in V[1:]])
        # self.ubs = ubs = np.array([1] + [n for j in V[1:]])

        # creating variables *x*
        self.x = x = np.empty((len(T), len(T), len(S)), dtype=Var)
        for i, j in permutations(range(len(T)), r=2):
            for k in range(len(S)):
                x[i, j, k] = model.add_var(f"x({i},{j},{k})", lb=0, ub=1, obj=0, var_type=INTEGER)

        # creating variables *b*
        self.bH = bH = []
        self.bA = bA = []
        for i in range(len(T)):
            for k in range(len(S)):
                bH.append(model.add_var(f"bH({i},{k})", lb=0, ub=1, obj=0, var_type=INTEGER))
                bA.append(model.add_var(f"bA({i},{k})", lb=0, ub=1, obj=0, var_type=INTEGER))

        # creating variables *s*
        self.s = s = []
        for c in range(len(C)):
            penalty = int(C[c].data["penalty"])
            if C[c].data["type"] == "HARD":
                penalty *= 1000000
            if C[c].tag == "CA1":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                for i in teams:
                    s.append(model.add_var(f"s({c},{i})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "CA2":
                teams1 = C[c].data["teams1"].split(";")
                teams1 = [int(x) for x in teams1]
                teams1.sort()
                for i in teams1:
                    s.append(model.add_var(f"sCA2({c},{i})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "CA3":
                teams1 = C[c].data["teams1"].split(";")
                teams1 = [int(x) for x in teams1]
                for i in teams1:
                    for k in range(len(S) - int(C[c].data["intp"]) + 1):
                        s.append(model.add_var(f"s({c},{i},{k})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "CA4" and C[c].data["mode2"] == "GLOBAL":
                s.append(model.add_var(f"sCA4({c})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "CA4" and C[c].data["mode2"] == "EVERY":
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                slots.sort()
                for k in slots:
                    s.append(model.add_var(f"sCA4({c},{k})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "GA1":
                s.append(model.add_var(f"s({c})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "BR1":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                for i in teams:
                    s.append(model.add_var(f"sBR1({c},{i})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "BR2":
                s.append(model.add_var(f"s({c})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "FA2":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                for i in teams:
                    for j in teams:
                        if i != j:
                            s.append(model.add_var(f"sFA2({c},{i},{j})", lb=0, obj=penalty, var_type=INTEGER))
            if C[c].tag == "SE1":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                min = int(C[c].data["min"])
                for i in teams:
                    for j in teams:
                        if i != j:
                            for k in range(len(S)):
                                for k2 in range(len(S)):
                                    if k > k2:
                                        diff = (min - (k - k2 - 1))
                                        if diff > 0:
                                            s.append(model.add_var(f"s({c},{i},{j},{k},{k2})", lb=0, ub=1, obj=penalty * diff, var_type=INTEGER))


        # creating constraints (1)
        for k in range(len(S)):
            for i in range(len(T)):
                model.add_constr(xsum((x[i, j, k] + x[j, i, k]) for j in range(len(T)) if j != i) == 1, name=f"cons1_{k}_{i}")

        # creating constraints (2)
        for i, j in permutations(range(len(T)), r=2):
            model.add_constr(xsum(x[i, j, k] for k in range(len(S))) == 1, name=f"cons2_{i}_{j}")

        # creating constraints (3)
        if self.inst.game_mode == "P":
            for k in range(int(len(S) / 2)):
                for i in range(len(T)):
                    for j in range(len(T)):
                        if i != j:
                            model.add_constr(x[i, j, k] + xsum(x[j, i, k2] for k2 in range(int(len(S) / 2)) if k != k2) <= 1, name=f"cons3_{k}_{i}_{j}")

        # creating constraints (4) and (5) CA1
        for c in range(len(C)):
            if C[c].tag == "CA1":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                for i in teams:
                    if C[c].data["mode"] == "H":
                        model.add_constr(xsum(x[i, j, k] for j in range(len(T)) if j != i for k in slots) <= int(C[c].data["max"]) + model.var_by_name(f"s({c},{i})"), name=f"cons4_{c}_{i}")
                    elif C[c].data["mode"] == "A":
                        model.add_constr(xsum(x[j, i, k] for j in range(len(T)) if j != i for k in slots) <= int(C[c].data["max"]) + model.var_by_name(f"s({c},{i})"), name=f"cons5_{c}_{i}")

        # creating constraints (15), (16) and (17) CA2
        for c in range(len(C)):
            if C[c].tag == "CA2":
                teams1 = C[c].data["teams1"].split(";")
                teams1 = [int(x) for x in teams1]
                teams1.sort()
                teams2 = C[c].data["teams2"].split(";")
                teams2 = [int(x) for x in teams2]
                teams2.sort()
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                slots.sort()
                max = int(C[c].data["max"])
                for i in teams1:
                    if C[c].data["mode1"] == "H":
                        model.add_constr(xsum(x[i, j, k] for k in slots for j in teams2 if i != j) <= max + model.var_by_name(f"sCA2({c},{i})"), name=f"cons15_{c}_{i}")
                    elif C[c].data["mode1"] == "A":
                        model.add_constr(xsum(x[j, i, k] for k in slots for j in teams2 if i != j) <= max + model.var_by_name(f"sCA2({c},{i})"), name=f"cons16_{c}_{i}")
                    elif C[c].data["mode1"] == "HA":
                        model.add_constr(xsum((x[i, j, k] + x[j, i, k]) for k in slots for j in teams2 if i != j) <= max + model.var_by_name(f"sCA2({c},{i})"), name=f"cons17_{c}_{i}")

        # creating constraints (6), (7) and (8) CA3
        for c in range(len(C)):
            if C[c].tag == "CA3":
                teams1 = C[c].data["teams1"].split(";")
                teams1 = [int(x) for x in teams1]
                teams2 = C[c].data["teams2"].split(";")
                teams2 = [int(x) for x in teams2]
                intp = int(C[c].data["intp"])
                max = int(C[c].data["max"])
                mode1 = C[c].data["mode1"]
                for i in teams1:
                    for k in range(len(S) - intp + 1):
                        if mode1 == "H":
                            model.add_constr(xsum(x[i, j, k + l] for l in range(intp) for j in teams2 if j != i) <= max + model.var_by_name(f"s({c},{i},{k})"), name=f"cons6_{c}_{i}_{k}")
                        elif mode1 == "A":
                            model.add_constr(xsum(x[j, i, k + l] for l in range(intp) for j in teams2 if j != i) <= max + model.var_by_name(f"s({c},{i},{k})"), name=f"cons6_{c}_{i}_{k}")
                        elif mode1 == "HA":
                            model.add_constr(xsum((x[i, j, k + l] + x[j, i, k + l]) for l in range(intp) for j in teams2 if j != i) <= max + model.var_by_name(f"s({c},{i},{k})"), name=f"cons6_{c}_{i}_{k}")

        # creating constraints (9) and (10) GA1
        for c in range(len(C)):
            if C[c].tag == "GA1":
                meetings = C[c].data["meetings"].split(";")
                meetings.remove("")
                meetings = [x.split(",") for x in meetings]
                for i in range(len(meetings)):
                    meetings[i] = (int(meetings[i][0]), int(meetings[i][1]))
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                min = int(C[c].data["min"])
                max = int(C[c].data["max"])
                if max < len(meetings):
                    model.add_constr(xsum(x[i, j, k] for (i, j) in meetings for k in slots) <= max + model.var_by_name(f"s({c})"), name=f"cons9_{c}")
                if min > 0:
                    model.add_constr(xsum(x[i, j, k] for (i, j) in meetings for k in slots) >= min - model.var_by_name(f"s({c})"), name=f"cons10_{c}")

        # creating constraints (11) and (12)
        for i in range(len(T)):
            for k in range(len(S) - 1):
                k2 = k + 1
                model.add_constr(xsum(x[i, j, k] for j in range(len(T)) if i != j) + xsum(x[i, j2, k2] for j2 in range(len(T)) if i != j2)
                                 <= 1 + model.var_by_name(f"bH({i},{k2})"), name=f"cons11_{i}_{k}")
                model.add_constr(xsum(x[j, i, k] for j in range(len(T)) if i != j) + xsum(x[j2, i, k2] for j2 in range(len(T)) if i != j2)
                                 <= 1 + model.var_by_name(f"bA({i},{k2})"), name=f"cons12_{i}_{k}")

        # creating constraints (18), (19) and (20) BR1
        for c in range(len(C)):
            if C[c].tag == "BR1":
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                slots.sort()
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                mode = C[c].data["mode2"]
                intp = int(C[c].data["intp"])
                for i in teams:
                    if mode == "H":
                        model.add_constr(xsum(model.var_by_name(f"bH({i},{k})") for k in slots) <= intp + model.var_by_name(f"sBR1({c},{i})"), name=f"cons18_{c}_{i}")
                    elif mode == "A":
                        model.add_constr(xsum(model.var_by_name(f"bH({i},{k})") for k in slots) <= intp + model.var_by_name(f"s19({c},{i})"), name=f"consBR1_{c}_{i}")
                    elif mode == "HA":
                        model.add_constr(xsum(model.var_by_name(f"bH({i},{k})") + model.var_by_name(f"bA({i},{k})") for k in slots) <= intp + model.var_by_name(f"sBR1({c},{i})"), name=f"cons20_{c}_{i}")

        # creating constraints (13) BR2
        for c in range(len(C)):
            if C[c].tag == "BR2":
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                slots.sort()
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                intp = int(C[c].data["intp"])
                model.add_constr(xsum(model.var_by_name(f"bH({i},{k})") + model.var_by_name(f"bA({i},{k})") for i in teams for k in slots) <= intp + model.var_by_name(f"s({c})"), name=f"cons13_{c}")

        # # creating constraints (14)
        # for c in range(len(C)):
        #     if C[c].tag == "SE1":
        #         teams = C[c].data["teams"].split(";")
        #         teams = [int(x) for x in teams]
        #         for i in teams:
        #             for j in teams:
        #                 if i != j:
        #                     model.add_constr(xsum((x[i,j,k] * (k + 1)) for k in slots) == y[i,j], name=f"cons14_{c}_{i}_{j}")
        #
        # # creating constraints (15) SE1
        # for c in range(len(C)):
        #     if C[c].tag == "SE1":
        #         teams = C[c].data["teams"].split(";")
        #         teams = [int(x) for x in teams]
        #         min = int(C[c].data["min"])
        #         for i in teams:
        #             for j in teams:
        #                 if i != j:
        #                     if C[c].data["type"] == "HARD":
        #                         model.add_constr(y[i,j] - y[j,i] >= min, name=f"cons15_{c}_{i}_{j}")
        #                     elif C[c].data["type"] == "SOFT":
        #                         model.add_constr(y[i,j] - y[j,i] >= min - model.var_by_name(f"s({c},{i},{j})"), name=f"cons15_{c}_{i}_{j}")

        # creating constraints (14)
        for c in range(len(C)):
            if C[c].tag == "SE1":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                min = int(C[c].data["min"])
                for i in teams:
                    for j in teams:
                        if i != j:
                            for k in range(len(S)):
                                for k2 in range(len(S)):
                                    if k > k2:
                                        if (min - (k - k2 - 1)) > 0:
                                            model.add_constr(x[i,j,k] + x[j,i,k2] <= 1 + model.var_by_name(f"s({c},{i},{j},{k},{k2})"), name=f"cons14_{c}_{i}_{j}_{k}_{k2}")

        # creating constraints (18), (19) and (20) BR1
        for c in range(len(C)):
            if C[c].tag == "FA2":
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                slots.sort()
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                mode = C[c].data["mode"]
                intp = int(C[c].data["intp"])
                for i in teams:
                    for j in teams:
                        if i != j:
                            if mode == "H":
                                model.add_constr(xsum(x[i, j2, k] for j2 in range(len(T)) if i != j2 for k in slots) - xsum(x[j, i2, k] for i2 in range(len(T)) if j != i2 for k in slots) <= intp + model.var_by_name(f"sFA2({c},{i},{j})"), name=f"cons21_{c}_{i}_{j}")

        # creating constraints (22) - (27) CA4
        for c in range(len(C)):
            if C[c].tag == "CA4":
                teams1 = C[c].data["teams1"].split(";")
                teams1 = [int(x) for x in teams1]
                teams1.sort()
                teams2 = C[c].data["teams2"].split(";")
                teams2 = [int(x) for x in teams2]
                teams2.sort()
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                slots.sort()
                max = int(C[c].data["max"])
                if C[c].data["mode2"] == "GLOBAL":
                    if C[c].data["mode1"] == "H":
                        model.add_constr(xsum(x[i, j, k] for i in teams1 for k in slots for j in teams2 if i != j) <= max + model.var_by_name(f"sCA4({c})"), name=f"cons22_{c}")
                    elif C[c].data["mode1"] == "A":
                        model.add_constr(xsum(x[j, i, k] for i in teams1 for k in slots for j in teams2 if i != j) <= max + model.var_by_name(f"sCA4({c})"), name=f"cons23_{c}")
                    elif C[c].data["mode1"] == "HA":
                        model.add_constr(xsum((x[i, j, k] + x[j, i, k]) for i in teams1 for k in slots for j in teams2 if i != j) <= max + model.var_by_name(f"sCA4({c})"), name=f"cons24_{c}")
                elif C[c].data["mode2"] == "EVERY":
                    for k in slots:
                        if C[c].data["mode1"] == "H":
                            model.add_constr(xsum(x[i, j, k] for i in teams1 for j in teams2 if i != j) <= max + model.var_by_name(f"sCA4({c},{k})"), name=f"cons25_{c}")
                        elif C[c].data["mode1"] == "A":
                            model.add_constr(xsum(x[j, i, k] for i in teams1 for j in teams2 if i != j) <= max + model.var_by_name(f"sCA4({c},{k})"), name=f"cons26_{c}")
                        elif C[c].data["mode1"] == "HA":
                            model.add_constr(xsum((x[i, j, k] + x[j, i, k]) for i in teams1 for j in teams2 if i != j) <= max + model.var_by_name(f"sCA4({c},{k})"), name=f"cons27_{c}")


    def set_solver_options(self):
        """Sets solver options"""

        # emphasis on optimally solving the model
        self.model.emphasis = self.params.emphasis
        self.model.max_seconds = self.params.timelimit
        self.model.threads = self.params.threads
        # keeps on printing
        self.model.verbose = 1

    def print_vars(self):
        if self.model.num_solutions == 0:
            return
        # Print active variables (for debug)
        for i in range(len(self.inst.teams)):
            for j in range(len(self.inst.teams)):
                if i != j:
                    for k in range(len(self.inst.slots)):
                        if self.x[i, j, k].x >= 0.00001:
                            print("%s \t %.3f" % (self.x[i, j, k].name, self.x[i, j, k].x))
        print()
        for i in self.s:
            if i.x >= 0.00001:
                print("%s \t %.3f" % (i.name, i.x))

        # for i in self.bH:
        #     if i.x >= 0.00001:
        #         print("%s \t %.3f" % (i.name, i.x))
        # for i in self.bA:
        #     if i.x >= 0.00001:
        #         print("%s \t %.3f" % (i.name, i.x))


    def optimize(self, max_seconds: int = 0):
        if max_seconds:
            self.model.optimize(max_seconds=max_seconds)
        else:
            self.model.optimize()


# #fix all x vars except the ones envolving teams in Tline
# for i in range(len(self.inst.teams)):
#     for j in range(len(self.inst.teams)):
#         if i != j:
#             for k in range(len(self.inst.slots)):
#                 if (self.x[i,j,k].x != None and self.x[i,j,k].x >= 0.999) or (i, j, k) in self.soln:
#                     self.x[i,j,k].lb = self.x[i,j,k].ub = 1
#                 else:
#                     self.x[i,j,k].lb = self.x[i,j,k].ub = 0
# #unfix vars
# for t in Tline:
#     for i in range(len(self.inst.teams)):
#         for j in range(len(self.inst.teams)):
#             if i != j and (t == i or t == j):
#                 for k in range(len(self.inst.slots)):
#                     self.x[i, j, k].lb = 0
#                     self.x[i, j, k].ub = 1
