from instance import Instance
from params import Parameters
import numpy as np
from mip import Var, Model, BINARY, INTEGER, CONTINUOUS, xsum, OptimizationStatus, CBC
from itertools import permutations
import random
import time
from solution import Solution
import itertools


class STTModel:
    __slots__ = ['inst', 'params', 'soln', 'ub', 'phase', 'is_optimal', 'model', 'x', 'bH', 'bA',
                 'sCA1', 'sCA2', 'sCA3', 'sCA4', 'sGA1', 'sBR1', 'sBR2', 'sFA2', 'sSE1']

    def __init__(self, inst: Instance, params: Parameters, phase=1, soln=None):
        self.inst = inst
        self.params = params
        self.soln = soln
        self.ub = float("inf")
        self.phase = phase
        self.is_optimal = 0

        # model-related variables
        self.model = None
        self.x = self.bH = self.bA = None
        self.sCA1 = self.sCA2 = self.sCA3 = self.sCA4 = self.sGA1 = None
        self.sBR1 = self.sBR2 = self.sFA2 = self.sSE1 = None

        # creating model and writing it to disk
        self.create_model()
        # self.model.write(self.params.output + ".lp")
        # self.model.write("temp_v4.lp")

        # solving model
        if self.soln is not None:
            self.load_soln()
        if self.params.approach == "fixopt":
            self.fixopt()
        elif self.params.approach == "mathprog":
            self.mathprog()

    def fix_unfeas_vars(self):
        T, S, C = self.inst.teams, self.inst.slots, self.inst.constrs
        fix_count = 0
        for c in range(len(C)):
            if C[c].tag == "CA1" and C[c].data["type"] == "HARD":
                teams = C[c].data["teams"].split(";")
                teams = [int(x) for x in teams]
                teams.sort()
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                max = int(C[c].data["max"])
                mode = C[c].data["mode"]
                if len(teams) == 1 and max == 0:
                    i = teams[0]
                    for k in slots:
                        for j in range(len(T)):
                            if i != j:
                                if mode == "H":
                                    fix_count += 1
                                    self.x[i,j,k].ub = 0
                                if mode == "A":
                                    fix_count += 1
                                    self.x[j,i,k].ub = 0
            if C[c].tag == "CA2" and C[c].data["type"] == "HARD":
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
                    for j in teams2:
                        if i != j:
                            for k in slots:
                                if C[c].data["mode1"] == "H" and max == 0:
                                    fix_count += 1
                                    self.x[i,j,k].ub = 0
                                elif C[c].data["mode1"] == "A" and max == 0:
                                    fix_count += 1
                                    self.x[j,i,k].ub = 0
                                elif C[c].data["mode1"] == "HA" and max == 0:
                                    fix_count += 2
                                    self.x[i,j,k].ub = self.x[j,i,k].ub = 0

            if C[c].tag == "GA1" and C[c].data["type"] == "HARD":
                meetings = C[c].data["meetings"].split(";")
                meetings.remove("")
                meetings = [x.split(",") for x in meetings]
                for i in range(len(meetings)):
                    meetings[i] = (int(meetings[i][0]), int(meetings[i][1]))
                slots = C[c].data["slots"].split(";")
                slots = [int(x) for x in slots]
                min = int(C[c].data["min"])
                max = int(C[c].data["max"])
                if min == len(meetings) and len(slots) == 1:
                    for (i, j) in meetings:
                        k = slots[0]
                        fix_count += 1
                        self.x[i,j,k].lb = 1
                        for k2 in range(len(S)):
                            if k != k2:
                                fix_count += 1
                                self.x[i, j, k2].ub = 0
                        for j2 in range(len(T)):
                            if j2 != j and j2 != i:
                                fix_count += 1
                                self.x[i, j2, k].ub = 0
                        for i2 in range(len(T)):
                            if i2 != i and i2 != j:
                                fix_count += 1
                                self.x[i2, j, k].ub = 0

                if max == 0:
                    for (i, j) in meetings:
                        for k in slots:
                            fix_count += 1
                            self.x[i, j, k].ub = 0
        print(f"Preprocessing fixed {fix_count} vars.")

    def rollhorizon(self):
        #self.fix_unfeas_vars()
        T, S, num_x_vars = self.inst.teams, self.inst.slots, len(self.inst.teams) * (len(self.inst.teams) - 1) * len(self.inst.slots)
        S = [x for x in range(len(S))]
        #mipStart = []
        soln = []
        self.model.relax()
        window_size = 1
        first_slot = 0
        int_slots = []
        for i in range(first_slot, first_slot + window_size):
            int_slots.append(i)
        fix_slots = []
        rel_slots = list(set(S) - set(int_slots))
        print("/-------------------------------------------------\\")
        print("|    slot |    size |    time |      lb |      ub |")
        print("|-------------------------------------------------|")
        while first_slot + window_size <= len(S):
            for i in range(len(self.inst.teams)):
                for j in range(len(self.inst.teams)):
                    if i != j:
                        # for k in range(len(S)):
                        #     self.x[i, j, k].var_type = "C"
                        #     self.x[i, j, k].lb = 0
                        #     self.x[i, j, k].ub = 1
                        for k in int_slots:
                            self.x[i, j, k].var_type = "B"
                        if self.params.local_branching == 0:
                            for k in fix_slots:
                                if self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999:
                                    #mipStart.append((self.x[i, j, k], 1))
                                    self.x[i, j, k].lb = self.x[i, j, k].ub = 1
                                    for k2 in range(len(S)):
                                        if k != k2:
                                            self.x[i, j, k2].ub = self.x[i, j, k2].lb = 0
                                    for j2 in range(len(T)):
                                        if j2 != j and j2 != i:
                                            self.x[i, j2, k].ub = self.x[i, j2, k].lb = 0
                                    for i2 in range(len(T)):
                                        if i2 != i and i2 != j:
                                            self.x[i2, j, k].ub = self.x[i2, j, k].lb = 0
                                else:
                                    #mipStart.append((self.x[i, j, k], 0))
                                    self.x[i, j, k].lb = self.x[i, j, k].ub = 0
                        for k in rel_slots:
                            self.x[i, j, k].var_type = "C"
            #print(fix_slots, int_slots, rel_slots)
            self.set_solver_options()
            #self.model.start = mipStart

            lb_constr = None
            if self.params.local_branching == 1:
                # Local Branching
                lb_constr = self.model.add_constr(xsum(
                    (1 - self.x[i, j, k]) for i in range(len(T)) for j in range(len(T)) if j != i for k in
                    fix_slots if ((self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999))) <= self.params.local_branching_k,
                                                  name=f"local_branching")
            #input("wait")
            print("| %7d | %7d |" % (first_slot, window_size), end="", flush=True)
            self.model.optimize(max_seconds=self.params.timelimit)
            for k in range(len(S)):
                if k in int_slots:
                    for i in range(len(T)):
                        for j in range(len(T)):
                            if i != j and self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999:
                                soln.append((i, j, k))
                                #print(i, j, k)
            if self.model.status == OptimizationStatus.INFEASIBLE:
                print(" %7d | %7s | %7s |" % ((time.time() - self.params.starttime), "inf", "inf"))
                inf_constr = self.model.add_constr(xsum(
                    self.x[i, j, k] for i in range(len(T)) for j in range(len(T)) for k in fix_slots if j != i and (i, j, k) in soln) <= (int(len(T) / 2) * len(fix_slots)) - 1, name=f"inf_constr")
                self.model.write(self.params.instance_file + ".lp")
                # #restart
                soln = []
                first_slot = 0
                int_slots = []
                for i in range(first_slot, first_slot + window_size):
                    int_slots.append(i)
                fix_slots = []
                rel_slots = list(set(S) - set(int_slots))
                # for k in range(first_slot - window_size, first_slot):
                #     fix_slots.remove(k)
                # first_slot -= window_size
                # # fix_slots += [first_slot]
                # # first_slot += 1
                # int_slots = []
                # for i in range(first_slot, first_slot + window_size):
                #     if i < len(S):
                #         int_slots.append(i)
                # rel_slots = list(set(S) - set(fix_slots) - set(int_slots))
                # print(fix_slots, int_slots, rel_slots)

                input("wait")
                continue
                # for k in int_slots:
                #     rel_slots.append(k)
                #     rel_slots.append(k-1)
                #     rel_slots.append(k-2)
                #     rel_slots.append(k-3)
                #     rel_slots.append(k-4)
                #     if k - 1 in fix_slots:
                #         fix_slots.remove(k - 1)
                #     if k - 2 in fix_slots:
                #         fix_slots.remove(k - 2)
                #     if k - 3 in fix_slots:
                #         fix_slots.remove(k - 3)
                #     if k - 4 in fix_slots:
                #         fix_slots.remove(k - 4)
                #     for i in range(len(self.inst.teams)):
                #         for j in range(len(self.inst.teams)):
                #             if i != j:
                #                 self.x[i, j, k].var_type = "C"
                #                 self.x[i, j, k-1].var_type = "C"
                #                 self.x[i, j, k - 2].var_type = "C"
                #                 self.x[i, j, k - 3].var_type = "C"
                #                 self.x[i, j, k - 4].var_type = "C"
                # if self.params.local_branching == 1:
                #     self.model.remove(lb_constr)
                #
                # #for k in range(first_slot, first_slot + window_size):
                # #    fix_slots.append(k)
                # first_slot += window_size
                # # fix_slots += [first_slot]
                # # first_slot += 1
                # int_slots = []
                # for i in range(first_slot, first_slot + window_size):
                #     if i < len(S):
                #         int_slots.append(i)
                # rel_slots = list(set(rel_slots) - set(int_slots))
            else:
                print(" %7d | %7d | %7d |" % ((time.time() - self.params.starttime), self.model.objective_bound, self.model.objective_value), end="")
                #self.print_vars()
                if self.params.local_branching == 1:
                    self.model.remove(lb_constr)

                for k in range(first_slot, first_slot + window_size):
                    fix_slots.append(k)
                first_slot += window_size
                # fix_slots += [first_slot]
                # first_slot += 1
                int_slots = []
                for i in range(first_slot, first_slot + window_size):
                    if i < len(S):
                        int_slots.append(i)
                rel_slots = list(set(rel_slots) - set(int_slots))
            print(fix_slots, int_slots, rel_slots)
        if self.model.num_solutions > 0 and self.model.objective_value < self.ub:
            self.ub = self.model.objective_value
            self.convert_soln()
            soln = None
            if self.phase == 1:
                soln = Solution(self.soln, hard_cost=self.ub)
            else:
                soln = Solution(self.soln, hard_cost=0, soft_cost=self.ub)
            soln.write(self.params)

    def mathprog(self):
        self.set_solver_options()
        #self.model.write(self.params.instance_file + ".lp")
        #self.model.clique_merge(self.model.constrs)
        self.model.optimize(max_seconds=self.params.timelimit)
        self.print_vars()
        if self.model.num_solutions > 0 and self.model.objective_value < self.ub:
            self.ub = self.model.objective_value
            self.convert_soln()
            soln = None
            if self.phase == 1:
                soln = Solution(self.soln, hard_cost=self.ub)
            else:
                soln = Solution(self.soln, hard_cost=0, soft_cost=self.ub)
            soln.write(self.params)
            if self.model.status == OptimizationStatus.OPTIMAL:
                print("Optimal solution found!")
                self.is_optimal = 1

    def fixopt(self):
        T, S, num_x_vars = self.inst.teams, self.inst.slots, len(self.inst.teams) * (len(self.inst.teams) - 1) * len(
            self.inst.slots)
        n = self.params.fixopt_n
        max_opt_in_row = 3
        opt_in_row = 0
        it = 0
        forbid_soln_constr = []
        print("/-----------------------------------------------------------\\")
        print("|      it |       n |    vars |    time |      lb |      ub |")
        print("|-----------------------------------------------------------|")
        Sline = []
        while time.time() - self.params.starttime < self.params.timelimit:
            it += 1
            free_vars = 0
            # mipStart = []
            # select vars to be unfixed
            current = random.choice(["t", "s", "s"])
            if self.params.fixopt_neighborhood == "t" or (self.params.fixopt_neighborhood == "all" and current == "t"):
                Tline = []
                Taux = [x for x in range(len(T))]
                if n >= len(T) - 1:
                    Tline = Taux
                else:
                    while len(Tline) < n / 2:
                        t = random.choice(Taux)
                        Taux.remove(t)
                        Tline.append(t)
                # fix all x vars except the selected ones (teams in Tline)
                for i in range(len(self.inst.teams)):
                    for j in range(len(self.inst.teams)):
                        if i != j:
                            for k in range(len(self.inst.slots)):
                                if i in Tline or j in Tline:
                                    free_vars += 1
                                    self.x[i, j, k].lb = 0
                                    self.x[i, j, k].ub = 1
                                else:
                                    if (self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999) or (
                                            (i, j, k) in self.soln and it == 1):
                                        self.x[i, j, k].lb = self.x[i, j, k].ub = 1
                                    else:
                                        self.x[i, j, k].lb = self.x[i, j, k].ub = 0
            elif self.params.fixopt_neighborhood == "p" or (
                    self.params.fixopt_neighborhood == "all" and current == "p"):
                if it == 1:
                    for i in range(len(self.inst.teams)):
                        for j in range(len(self.inst.teams)):
                            if i != j:
                                for k in range(len(self.inst.slots)):
                                    if (i, j, k) in self.soln:
                                        self.x[i, j, k].lb = self.x[i, j, k].ub = 1
                                    else:
                                        self.x[i, j, k].lb = self.x[i, j, k].ub = 0
                else:
                    if it == 2:
                        for i in range(len(self.inst.teams)):
                            for j in range(len(self.inst.teams)):
                                if i != j:
                                    for k in range(len(self.inst.slots)):
                                        self.x[i, j, k].lb = 0
                                        self.x[i, j, k].ub = 1
                    prob = 0.05
                    num_x_vars = len(self.sCA1) + len(self.sCA2) + len(self.sCA3) + len(self.sCA4) + len(
                        self.sGA1) + len(self.sBR1) + len(self.sBR2) + len(self.sFA2) + len(self.sSE1)
                    print(num_x_vars)
                    for c, i in self.sCA1:
                        if random.random() < prob or self.sCA1[c, i].x is None:
                            self.sCA1[c, i].lb = 0
                            self.sCA1[c, i].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sCA1[c, i].lb = self.sCA1[c, i].ub = self.sCA1[c, i].x
                    for c, i in self.sCA2:
                        if random.random() < prob or self.sCA2[c, i].x is None:
                            self.sCA2[c, i].lb = 0
                            self.sCA2[c, i].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sCA2[c, i].lb = self.sCA2[c, i].ub = self.sCA2[c, i].x
                    for c, i, k in self.sCA3:
                        if random.random() < prob or self.sCA3[c, i, k].x is None:
                            self.sCA3[c, i, k].lb = 0
                            self.sCA3[c, i, k].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sCA3[c, i, k].lb = self.sCA3[c, i, k].ub = self.sCA3[c, i, k].x
                    for c, k in self.sCA4:
                        if random.random() < prob or self.sCA4[c, k].x is None:
                            self.sCA4[c, k].lb = 0
                            self.sCA4[c, k].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sCA4[c, k].lb = self.sCA4[c, k].ub = self.sCA4[c, k].x
                    for c in self.sGA1:
                        if random.random() < prob or self.sGA1[c].x is None:
                            self.sGA1[c].lb = 0
                            self.sGA1[c].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sGA1[c].lb = self.sGA1[c].ub = self.sGA1[c].x
                    for c, i in self.sBR1:
                        if random.random() < prob or self.sBR1[c, i].x is None:
                            self.sBR1[c, i].lb = 0
                            self.sBR1[c, i].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sBR1[c, i].lb = self.sBR1[c, i].ub = self.sBR1[c, i].x
                    for c in self.sBR2:
                        if random.random() < prob or self.sBR2[c].x is None:
                            self.sBR2[c].lb = 0
                            self.sBR2[c].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sBR2[c].lb = self.sBR2[c].ub = self.sBR2[c].x
                    for c, i, j, k in self.sFA2:
                        if random.random() < prob or self.sFA2[c, i, j, k].x is None:
                            self.sFA2[c, i, j, k].lb = 0
                            self.sFA2[c, i, j, k].ub = float("inf")
                        else: #fix var
                            free_vars += 1
                            self.sFA2[c, i, j, k].lb = self.sFA2[c, i, j, k].ub = self.sFA2[c, i, j, k].x
                    for i, j, k, k2 in self.sSE1:
                        if random.random() < prob or self.sSE1[i, j, k, k2].x is None:
                            self.sSE1[i, j, k, k2].lb = 0
                            self.sSE1[i, j, k, k2].ub = 1
                        else: #fix var
                            free_vars += 1
                            self.sSE1[i, j, k, k2].lb = self.sSE1[i, j, k, k2].ub = self.sSE1[i, j, k, k2].x
            elif self.params.fixopt_neighborhood == "s" or (
                    self.params.fixopt_neighborhood == "all" and current == "s"):
                x_free = np.zeros((len(T), len(T), len(S)), dtype=bool)
                Saux = [x for x in range(len(S))]
                Sline = []
                if n >= len(S):
                    Sline = Saux
                else:
                    while len(Sline) < n:
                        k = random.choice(Saux)
                        # random
                        Saux.remove(k)
                        Sline.append(k)
                        # and next
                        if len(Sline) < n and k + 1 < len(S) and k + 1 in Saux:
                            Saux.remove(k + 1)
                            Sline.append(k + 1)
                        # and previous
                        # if len(Sline) < n and k - 1 > 0 and k - 1 in Saux:
                        #     Saux.remove(k - 1)
                        #     Sline.append(k - 1)

                        # sliding window
                        # for i in range(k, k + n):
                        #     # Saux.remove(k)
                        #     if i >= len(S):
                        #         break
                        #     Sline.append(i)
                        # if len(Sline) < n:
                        #     Sline += Saux[0:n-len(Sline)]
                # print(Sline)
                #Sline = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                for i in range(len(self.inst.teams)):
                    for j in range(len(self.inst.teams)):
                        if i != j:
                            for k in range(len(self.inst.slots)):
                                if k in Sline:
                                    x_free[i, j, k] = 1
                                    if (self.x[i, j, k].x != None and (self.x[i, j, k].x >= 0.999)) or (
                                            it == 1 and ((i, j, k) in self.soln)):  # (i,j) occurs in k
                                        for k2 in range(len(self.inst.slots)):
                                            if k != k2 and (
                                                    (self.x[j, i, k2].x != None and (self.x[j, i, k2].x >= 0.999)) or (
                                                    it == 1 and ((j, i, k2) in self.soln))):  # (j,i) occurs in k2
                                                x_free[j, i, k2] = 1
                                                x_free[i, j, k2] = 1
                # fix all x vars except the selected ones (teams in Tline)
                for i in range(len(self.inst.teams)):
                    for j in range(len(self.inst.teams)):
                        if i != j:
                            for k in range(len(self.inst.slots)):
                                if x_free[i, j, k] == 1:
                                    free_vars += 1
                                    self.x[i, j, k].lb = 0
                                    self.x[i, j, k].ub = 1
                                else:
                                    if (self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999) or (
                                            it == 1 and (i, j, k) in self.soln):
                                        # mipStart.append((self.x[i, j, k], 1))
                                        self.x[i, j, k].lb = self.x[i, j, k].ub = 1
                                    else:
                                        # mipStart.append((self.x[i, j, k], 0))
                                        self.x[i, j, k].lb = self.x[i, j, k].ub = 0
            elif self.params.fixopt_neighborhood == "m" or (
                    self.params.fixopt_neighborhood == "all" and current == "m"):
                # fix n matches for each slot
                for k in range(len(self.inst.slots)):
                    countN = 0
                    nInv = int(len(self.inst.teams) / 2) - int(n / 2)
                    t1 = [x for x in range(len(self.inst.teams))]
                    random.shuffle(t1)
                    t2 = [x for x in range(len(self.inst.teams))]
                    random.shuffle(t2)
                    for i in t1:
                        for j in t2:
                            if i != j:
                                free_vars += 1
                                self.x[i, j, k].lb = 0
                                self.x[i, j, k].ub = 1
                                if ((self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999) or (
                                        (i, j, k) in self.soln and it == 1)) and countN < nInv:
                                    # fix var
                                    self.x[i, j, k].lb = self.x[i, j, k].ub = 1
                                    countN += 1
                                    free_vars -= 1
                                    for j2 in t2:
                                        if j2 != j and j2 != i:
                                            self.x[i, j2, k].lb = self.x[i, j2, k].ub = 0
                                            free_vars -= 1
                                    for k2 in range(len(self.inst.slots)):
                                        if k2 != k:
                                            self.x[i, j, k2].lb = self.x[i, j, k2].ub = 0
                                            free_vars -= 1
            # self.model.start = mipStart
            if self.params.fixopt_neighborhood == "s" or (self.params.fixopt_neighborhood == "all" and current == "s"):
                print("| %7d | %6ds | %7d |" % (it, n, free_vars), end="", flush=True)
            elif self.params.fixopt_neighborhood == "t" or (self.params.fixopt_neighborhood == "all" and current == "t"):
                print("| %7d | %6dt | %7d |" % (it, n, free_vars), end="", flush=True)
            elif self.params.fixopt_neighborhood == "p" or (self.params.fixopt_neighborhood == "all" and current == "p"):
                print("| %7d | %6dt | %7d |" % (it, n, free_vars), end="", flush=True)

            # solve model
            # self.model.write(self.params.instance_file + ".lp")
            self.set_solver_options()
            max_time = min(self.params.fixopt_timelimit, self.params.timelimit - (time.time() - self.params.starttime))
            self.model.optimize(max_seconds=max_time)
            runtime = time.time() - self.params.starttime

            if self.model.num_solutions == 0 or self.model.objective_value >= self.ub * 1.1:
                if self.params.fixopt_neighborhood == "s" or (
                        self.params.fixopt_neighborhood == "all" and current == "s"):
                    print(f" {runtime:7.0f} |    None |    None | {Sline}", flush=True)
                # retrieve best soln and restart search
                print("New solution is too bad.. best solution retrieved!")
                for const in forbid_soln_constr:
                    self.model.remove(const)
                forbid_soln_constr = []
                self.load_soln()
                it = 0
                continue

            if self.params.fixopt_neighborhood == "s" or (self.params.fixopt_neighborhood == "all" and current == "s"):
                print(f" {runtime:7.0f} | {self.model.objective_bound:7.0f} | {self.model.objective_value:7.0f} | {Sline}", flush=True)
            elif self.params.fixopt_neighborhood == "t" or (self.params.fixopt_neighborhood == "all" and current == "t"):
                print(f" {runtime:7.0f} | {self.model.objective_bound:7.0f} | {self.model.objective_value:7.0f} | {Tline}", flush=True)

            # auto-adaption of parameter n
            if self.model.status == OptimizationStatus.OPTIMAL and self.model.objective_value < self.ub:  # Optimal solution found (new best)
                pass
            elif self.model.status == OptimizationStatus.OPTIMAL:  # Optimal solution found (but not better than previous)
                opt_in_row += 1
                if opt_in_row >= max_opt_in_row:
                    opt_in_row = 0
                    n += 1
                    # forbids current solution
                    forbid_soln_constr.append(self.model.add_constr(xsum(
                        (self.x[i, j, k]) for i in range(len(T)) for j in range(len(T)) if j != i for k in
                        range(len(S)) if ((self.x[i, j, k].x != None and self.x[i, j, k].x >= 0.999))) <= (int(len(T) / 2) * len(S)) - 1,
                                                      name=f"forbid_soln_constr_{it}"))
                    print("Solution forbidden!")
                    # #self.model.write(self.params.instance_file + ".lp")
                    # input("wait")


            elif self.model.status != OptimizationStatus.OPTIMAL:  # Optimal solution has not been found (possible due to timelimit)
                n -= 1
                opt_in_row = 0

            # write solution if it is a new best
            if self.model.num_solutions > 0 and self.model.objective_value < self.ub:
                self.ub = self.model.objective_value
                self.convert_soln()
                soln = None
                if self.phase == 1:
                    soln = Solution(self.soln, hard_cost=self.ub)
                else:
                    soln = Solution(self.soln, hard_cost=0, soft_cost=self.ub)
                soln.write(self.params)
            if self.model.objective_value == 0 or (
                    self.model.status == OptimizationStatus.OPTIMAL and free_vars >= num_x_vars):
                print("\\-----------------------------------------------------------/")
                print("Optimal solution found!")
                self.is_optimal = 1
                return
        print("\\-----------------------------------------------------------/")

    def load_soln(self):
        mip_start = []
        for i in range(len(self.inst.teams)):
            for j in range(len(self.inst.teams)):
                if i != j:
                    for k in range(len(self.inst.slots)):
                        if (i, j, k) in self.soln:
                            mip_start.append((self.x[i, j, k], 1))
                            # self.x[i,j,k].lb = self.x[i,j,k].ub = 1
                        else:
                            mip_start.append((self.x[i, j, k], 0))
                            # self.x[i, j, k].lb = self.x[i, j, k].ub = 0
        self.model.start = mip_start

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
        inst, constrs = self.inst, self.inst.constrs
        teams, slots = [t.idx for t in self.inst.teams], [s.idx for s in self.inst.slots]
        games = list(permutations(teams, r=2))
        slot_pairs = [(k, k2) for k in slots for k2 in slots[k + 1:]]
        random.shuffle(constrs)

        # creating variables *x*
        self.x = x = np.empty((len(teams), len(teams), len(slots)), dtype=Var)
        for i, j in games:
            for k in slots:
                x[i, j, k] = model.add_var(f"x({i},{j},{k})", obj=0, var_type=BINARY)

        # creating variables *b*
        self.bH = bH = np.empty((len(teams), len(slots)), dtype=Var)
        self.bA = bA = np.empty((len(teams), len(slots)), dtype=Var)
        for i in teams:
            for k in slots:
                bH[i, k] = model.add_var(f"bH({i},{k})", obj=0, var_type=BINARY)
                bA[i, k] = model.add_var(f"bA({i},{k})", obj=0, var_type=BINARY)

        # creating containers for slack variables *s*
        self.sCA1 = sCA1 = {}
        self.sCA2 = sCA2 = {}
        self.sCA3 = sCA3 = {}
        self.sCA4 = sCA4 = {}
        self.sGA1 = sGA1 = {}
        self.sBR1 = sBR1 = {}
        self.sBR2 = sBR2 = {}
        self.sFA2 = sFA2 = {}
        self.sSE1 = sSE1 = {}#np.empty((len(teams), len(teams), len(slots), len(slots)), dtype=Var)

        # creating slack variables for selected constraints
        for c, constr in enumerate(constrs):
            # skipping all SOFT constraint if in Phase 1 (in Phase 1 SOFT constraints are ignored)
            if self.phase == 1 and constr.type == "SOFT":
                continue

            # setting s_ub to 0 if constraints is HARD in Phase 2 (in Phase 2 HARD constraints must be enforced)
            s_ub = float("inf")
            if self.phase == 2 and constr.type == "HARD":
                s_ub = 0

            if constr.tag == "CA1":
                for i in constr.teams:
                    sCA1[c, i] = model.add_var(f"sCA1({c},{i})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "CA2":
                for i in constr.teams1:
                    sCA2[c, i] = model.add_var(f"sCA2({c},{i})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "CA3":
                for i in constr.teams1:
                    for k in range(len(slots) - constr.intp + 1):
                        sCA3[c, i, k] = model.add_var(f"sCA3({c},{i},{k})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "CA4":
                if constr.mode2 == "GLOBAL":
                    sCA4[c, "GLOBAL"] = model.add_var(f"sCA4({c},GLOBAL)", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)
                elif constr.mode2 == "EVERY":
                    for k in constr.slots:
                        sCA4[c, k] = model.add_var(f"sCA4({c},{k})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "GA1":
                sGA1[c] = model.add_var(f"sGA1({c})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "BR1":
                for i in constr.teams:
                    sBR1[c, i] = model.add_var(f"sBR1({c},{i})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "BR2":
                sBR2[c] = model.add_var(f"sBR2({c})", lb=0, ub=s_ub, obj=constr.penalty, var_type=INTEGER)

            if constr.tag == "FA2":
                constr_games = permutations(constr.teams, r=2)
                for i, j in constr_games:
                    for k in constr.slots:
                        sFA2[c, i, j, k] = model.add_var(f"sFA2({c},{i},{j},{k})", lb=0, ub=s_ub,
                                                         obj=constr.penalty, var_type=INTEGER)
            if constr.tag == "SE1":
                for i, j in games:
                    for k, k2 in slot_pairs:
                        diff = 0
                        if i in constr.teams and j in constr.teams and constr.min - (k2 - k - 1) > 0:
                            diff = constr.min - (k2 - k - 1)
                        if type == "HARD" and diff > 0:
                            sSE1[i, j, k, k2] = model.add_var(f"sSE1({i},{j},{k},{k2})", lb=0, ub=0,
                                                              obj=constr.penalty * diff, var_type=BINARY)
                        else:
                            sSE1[i, j, k, k2] = model.add_var(f"sSE1({i},{j},{k},{k2})",
                                                              obj=constr.penalty * diff, var_type=BINARY)

        # creating constraints (1)
        for k in slots:
            for i in teams:
                model.add_constr(xsum((x[i, j, k] + x[j, i, k]) for j in teams if j != i) == 1,
                                 name=f"cons1_{k}_{i}")

        # creating constraints (2)
        for i, j in games:
            model.add_constr(xsum(x[i, j, k] for k in slots) == 1, name=f"cons2_{i}_{j}")

        # creating constraints (3)
        if self.inst.game_mode == "P":
            for k in range(int(len(slots) / 2)):
                for i, j in games:
                    model.add_constr(x[i, j, k] + xsum(x[j, i, k2] for k2 in range(int(len(slots) / 2)) if k != k2)
                                     <= 1, name=f"cons3_{k}_{i}_{j}")

        # creating constraints (11) and (12)
        for i in teams:
            for k in slots[:-1]:
                k2 = k + 1
                model.add_constr(xsum(x[i, j, k] for j in teams if i != j) +
                                 xsum(x[i, j, k2] for j in teams if i != j)
                                 <= 1 + bH[i, k2], name=f"cons11_{i}_{k}")
                model.add_constr(xsum(x[j, i, k] for j in teams if i != j) +
                                 xsum(x[j, i, k2] for j in teams if i != j)
                                 <= 1 + bA[i, k2], name=f"cons12_{i}_{k}")

        # loop to create instance-specific constraints
        for c, constr in enumerate(constrs):
            # creating constraints (4) and (5) | CA1
            if constr.tag == "CA1" and not (self.phase == 1 and constr.type == "SOFT"):
                for i in constr.teams:
                    if constr.mode == "H":
                        model.add_constr(xsum(x[i, j, k] for j in teams if j != i for k in constr.slots)
                                         <= constr.max + sCA1[c, i], name=f"cons4_{c}_{i}")
                    elif constr.mode == "A":
                        model.add_constr(xsum(x[j, i, k] for j in teams if j != i for k in constr.slots)
                                         <= constr.max + sCA1[c, i], name=f"cons5_{c}_{i}")

            # creating constraints (6), (7) and (8) | CA3
            if constr.tag == "CA3" and not (self.phase == 1 and constr.type == "SOFT"):
                for i in constr.teams1:
                    for k in range(len(slots) - constr.intp + 1):
                        if constr.mode1 == "H":
                            model.add_constr(xsum(x[i, j, k + l] for l in range(constr.intp)
                                                  for j in constr.teams2 if j != i)
                                             <= constr.max + sCA3[c, i, k], name=f"cons6_{c}_{i}_{k}")
                        elif constr.mode1 == "A":
                            model.add_constr(xsum(x[j, i, k + l] for l in range(constr.intp)
                                                  for j in constr.teams2 if j != i)
                                             <= constr.max + sCA3[c, i, k], name=f"cons7_{c}_{i}_{k}")
                        elif constr.mode1 == "HA":
                            model.add_constr(xsum((x[i, j, k + l] + x[j, i, k + l]) for l in range(constr.intp)
                                                  for j in constr.teams2 if j != i)
                                             <= constr.max + sCA3[c, i, k], name=f"cons8_{c}_{i}_{k}")

            # creating constraints (9) and (10) | GA1
            if constr.tag == "GA1" and not (self.phase == 1 and constr.type == "SOFT"):
                if constr.max < len(constr.meetings):
                    model.add_constr(xsum(x[i, j, k] for (i, j) in constr.meetings for k in constr.slots)
                                     <= constr.max + sGA1[c], name=f"cons9_{c}")
                if constr.min > 0:
                    model.add_constr(xsum(x[i, j, k] for (i, j) in constr.meetings for k in constr.slots)
                                     >= constr.min - sGA1[c], name=f"cons10_{c}")

            # creating constraints (13) | BR2
            if constr.tag == "BR2" and not (self.phase == 1 and constr.type == "SOFT"):
                model.add_constr(xsum(bH[i, k] + bA[i, k] for i in constr.teams for k in constr.slots)
                                 <= constr.intp + sBR2[c], name=f"cons13_{c}")

            # creating constraints (14) | SE1
            if constr.tag == "SE1" and not (self.phase == 1 and constr.type == "SOFT"):
                for i, j in games:
                    for k in slots:
                        for k2 in range(k + 1, len(slots)):
                            model.add_constr(x[i, j, k] + x[j, i, k2] <= 1 + sSE1[i, j, k, k2],
                                             name=f"cons14A_{i}_{j}_{k}_{k2}")
                #             model.add_constr(x[i, j, k] + x[j, i, k2] >= 2 * sSE1[i, j, k, k2],
                #                              name=f"cons14B_{i}_{j}_{k}_{k2}")
                #         model.add_constr(x[i, j, k] == xsum(sSE1[j, i, k2, k] for k2 in slots[:k]) +
                #                          xsum(sSE1[i, j, k, k2] for k2 in slots[k + 1:]),
                #                          name=f"cons14C_{i}_{j}_{k}")
                # for i in teams:
                #     for j in teams[i + 1:]:
                #         model.add_constr(xsum(sSE1[i, j, k, k2] + sSE1[j, i, k, k2] for k, k2 in slot_pairs)
                #                          == 1, name=f"cons14D_{i}_{j}")

            # creating constraints (15), (16) and (17) | CA2
            if constr.tag == "CA2" and not (self.phase == 1 and constr.type == "SOFT"):
                for i in constr.teams1:
                    if constr.mode1 == "H":
                        model.add_constr(xsum(x[i, j, k]
                                              for k in constr.slots
                                              for j in constr.teams2 if i != j)
                                         <= constr.max + sCA2[c, i], name=f"cons15_{c}_{i}")
                    elif constr.mode1 == "A":
                        model.add_constr(xsum(x[j, i, k]
                                              for k in constr.slots
                                              for j in constr.teams2 if i != j)
                                         <= constr.max + sCA2[c, i], name=f"cons16_{c}_{i}")
                    elif constr.mode1 == "HA":
                        model.add_constr(xsum((x[i, j, k] + x[j, i, k])
                                              for k in constr.slots
                                              for j in constr.teams2 if i != j)
                                         <= constr.max + sCA2[c, i], name=f"cons17_{c}_{i}")

            # creating constraints (18), (19) and (20) | BR1
            if constr.tag == "BR1" and not (self.phase == 1 and constr.type == "SOFT"):
                for i in constr.teams:
                    if constr.mode2 == "H":
                        model.add_constr(xsum(bH[i, k] for k in constr.slots) <= constr.intp + sBR1[c, i],
                                         name=f"cons18_{c}_{i}")
                    elif constr.mode2 == "A":
                        model.add_constr(xsum(bA[i, k] for k in constr.slots) <= constr.intp + sBR1[c, i],
                                         name=f"cons19_{c}_{i}")
                    elif constr.mode2 == "HA":
                        model.add_constr(xsum(bH[i, k] + bA[i, k] for k in constr.slots)
                                         <= constr.intp + sBR1[c, i], name=f"cons20_{c}_{i}")

            # creating constraints (21) | FA2
            if constr.tag == "FA2" and not (self.phase == 1 and constr.type == "SOFT"):
                if constr.mode == "H":
                    for i, j in permutations(constr.teams, r=2):
                        for k in constr.slots:
                            model.add_constr(xsum(x[i, j2, k2]
                                                  for j2 in teams if i != j2
                                                  for k2 in constr.slots if k2 <= k)
                                             - xsum(x[j, i2, k2]
                                                    for i2 in teams if j != i2
                                                    for k2 in constr.slots if k2 <= k)
                                             <= constr.intp + sFA2[c, i, j, k], name=f"cons21_{c}_{i}_{j}_{k}")

            # creating constraints (22)-(27) | CA4
            if constr.tag == "CA4" and not (self.phase == 1 and constr.type == "SOFT"):
                if constr.mode2 == "GLOBAL":
                    if constr.mode1 == "H":
                        model.add_constr(xsum(x[i, j, k]
                                              for i in constr.teams1
                                              for k in constr.slots
                                              for j in constr.teams2 if i != j)
                                         <= constr.max + sCA4[c, "GLOBAL"], name=f"cons22_{c}")
                    elif constr.mode1 == "A":
                        model.add_constr(xsum(x[j, i, k] for i in constr.teams1
                                              for k in constr.slots
                                              for j in constr.teams2 if i != j)
                                         <= constr.max + sCA4[c, "GLOBAL"], name=f"cons23_{c}")
                    elif constr.mode1 == "HA":
                        model.add_constr(xsum(x[i, j, k] + x[j, i, k] for i in constr.teams1
                                              for k in constr.slots
                                              for j in constr.teams2 if i != j)
                                         <= constr.max + sCA4[c, "GLOBAL"], name=f"cons24_{c}")

                elif constr.mode2 == "EVERY":
                    for k in constr.slots:
                        if constr.mode1 == "H":
                            model.add_constr(xsum(x[i, j, k] for i in constr.teams1
                                                  for j in constr.teams2 if i != j)
                                             <= constr.max + sCA4[c, k], name=f"cons25_{c}_{k}")
                        elif constr.mode1 == "A":
                            model.add_constr(xsum(x[j, i, k] for i in constr.teams1
                                                  for j in constr.teams2 if i != j)
                                             <= constr.max + sCA4[c, k], name=f"cons26_{c}_{k}")
                        elif constr.mode1 == "HA":
                            model.add_constr(xsum((x[i, j, k] + x[j, i, k]) for i in constr.teams1
                                                  for j in constr.teams2 if i != j)
                                             <= constr.max + sCA4[c, k], name=f"cons27_{c}_{k}")

    def set_solver_options(self):
        """Sets solver options"""
        # emphasis on optimally solving the model
        self.model.emphasis = self.params.emphasis
        self.model.max_seconds = self.params.timelimit
        self.model.threads = self.params.threads
        if self.params.seed != None:
            self.model.seed = self.params.seed
        # keeps on printing
        self.model.verbose = self.params.verbose
        # self.model.store_search_progress_log = True

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
        # for i in self.s:
        #     if i.x >= 0.00001:
        #         print("%s \t %.3f" % (i.name, i.x))

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
