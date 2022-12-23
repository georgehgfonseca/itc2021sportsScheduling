from instance import *
import xml.etree.ElementTree as ET
from instance import Instance
from params import Parameters
import time

class Solution:
    __slots__ = ['timetable', 'hard_cost', 'soft_cost']

    def __init__(self, timetable, hard_cost=float('inf'), soft_cost=float('inf')):
        self.timetable = timetable
        self.hard_cost = hard_cost
        self.soft_cost = soft_cost

    def eval(self) -> float:
        self.hard_cost = 0.0
        self.soft_cost = 0.0
        return 0.0

    def write2(self, xml_path):
        data = ET.Element('Solution')
        items = ET.SubElement(data, 'Games')
        for t in self.timetable:
            item1 = ET.SubElement(items, 'ScheduledMatch')
            item1.set('home', str(t[0]))
            item1.set('away', str(t[1]))
            item1.set('slot', str(t[2]))
        mydata = ET.tostring(data)
        myfile = open(xml_path, "wb")
        myfile.write(mydata)

    def write(self, params:Parameters):
        data = ET.Element('Solution')

        items = ET.SubElement(data, 'MetaData')
        item1 = ET.SubElement(items, 'InstanceName')
        item1.text = params.instance_file.replace("/", "_")
        item2 = ET.SubElement(items, 'SolutionName')
        item2.text = params.instance_file.replace("/", "_")
        item3 = ET.SubElement(items, 'ObjectiveValue')
        item3.set('infeasibility', str(self.hard_cost))
        item3.set('objective', str(self.soft_cost))
        item4 = ET.SubElement(items, 'Contributor')
        item4.text = "George Fonseca and Tulio Toffolo"
        # item5 = ET.SubElement(items, 'SolutionMethod')
        # item5.text = params.approach #bug in the validator!
        item6 = ET.SubElement(items, 'Date')
        item6.text = str(time.ctime())
        item7 = ET.SubElement(items, 'Remarks')
        item7.text = f"Solution method: {params.approach} Random seed: {params.seed} Timelimit: {params.timelimit} Threads: {params.threads}"

        items = ET.SubElement(data, 'Games')
        for t in self.timetable:
            item1 = ET.SubElement(items, 'ScheduledMatch')
            item1.set('home', str(t[0]))
            item1.set('away', str(t[1]))
            item1.set('slot', str(t[2]))
        mydata = ET.tostring(data)
        myfile = open(params.output, "wb")
        myfile.write(mydata)

    def read_xml_soln(xml_path):
        s = []
        inf = float("inf")
        obj = float("inf")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root:
            for subelem in elem:
                if "infeasibility" in subelem.attrib:
                    inf = float(subelem.attrib["infeasibility"])
                    obj = float(subelem.attrib["objective"])
                if "home" in subelem.attrib:
                    i = int(subelem.attrib["home"])
                    j = int(subelem.attrib["away"])
                    k = int(subelem.attrib["slot"])
                    s.append((i, j, k))
        return (s, inf, obj)

    def build_init_soln(inst: Instance):
        """"Polygon method to quickly build initial solutions"""
        inf = float("inf")
        obj = float("inf")
        soln = []
        T = [x for x in range(len(inst.teams))]
        S = [x for x in range(len(inst.slots))]
        Ttemp = T.copy()
        Ttemp = [int(x) for x in Ttemp]
        t0 = Ttemp[0]
        Ttemp.pop(0)
        for i in range(int(len(S) / 2)):
            if i % 2 == 0:
                soln.append((Ttemp[0], t0, i))
                soln.append((t0, Ttemp[0], i + int(len(S) / 2))) #mirroed
            else: #invert venue
                soln.append((t0, Ttemp[0], i))
                soln.append((Ttemp[0], t0, i + int(len(S) / 2))) #mirroed
            for j in range(1, int((len(Ttemp)) / 2) + 1):
                t1 = Ttemp[j]
                t2 = Ttemp[-j]
                if i % 2 == 0:
                    soln.append((t2, t1, i))
                    soln.append((t1, t2, i + int(len(S) / 2)))  # mirroed
                else: #invert venue
                    soln.append((t1, t2, i))
                    soln.append((t2, t1, i + int(len(S) / 2)))  # mirroed
            aux = Ttemp.pop(0)
            Ttemp.append(aux)
        return (soln, inf, obj)


