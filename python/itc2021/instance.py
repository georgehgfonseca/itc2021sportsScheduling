from constrs import *
import xml.etree.ElementTree as etree


class League:
    __slots__ = ['idx', 'name', 'constrs']

    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name

        # data to be preprocessed
        self.constrs = []


class Slot:
    __slots__ = ['idx', 'name', 'constrs']

    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name

        # data to be preprocessed
        self.constrs = []


class Team:
    __slots__ = ['idx', 'league', 'name', 'constrs']

    def __init__(self, idx: int, league: League, name: str):
        self.idx = idx
        self.league = league
        self.name = name

        # data to be preprocessed
        self.constrs = []


class Instance:
    __slots__ = [
        'xml', 'instance_name', 'description',
        'leagues', 'teams', 'slots',
        'number_rr', 'compactness', 'game_mode', 'objective',
        'constrs', 'hard_constrs', 'soft_constrs',
        'preprocessed'
    ]

    def __init__(self, xml_path):
        # initializing auxiliary variables
        self.preprocessed = False

        self.xml = xml = etree.parse(xml_path).getroot()

        # reading metadata
        self.instance_name = xml.find('./MetaData/InstanceName').text
        if xml.find('./MetaData/Description') != None:
            self.description = xml.find('./MetaData/Description').text

        # reading resources
        self.leagues = [League(int(it.attrib['id']), it.attrib['name'])
                        for it in xml.findall('./Resources/Leagues/league')]
        self.teams = [Team(int(it.attrib['id']), self.leagues[int(it.attrib['league'])], it.attrib['name'])
                      for it in xml.findall('./Resources/Teams/team')]
        self.slots = [Slot(int(it.attrib['id']), it.attrib['name'])
                      for it in xml.findall('./Resources/Slots/slot')]
        assert len(self.leagues) == 1, 'Too many leagues: the solver supports only one'
        assert len(self.slots) == (len(self.teams) - 1) * 2, 'Incorrect number of slots or teams'

        # reading structure/format
        self.number_rr = int(xml.find('./Structure/Format/numberRoundRobin').text)
        self.compactness = xml.find('./Structure/Format/compactness').text
        self.game_mode = xml.find('./Structure/Format/gameMode').text
        assert self.number_rr == 2, 'Wrong number of Round Robins'
        assert self.game_mode in ['P', 'NULL'], 'Invalid game mode'

        # reading objective
        self.objective = xml.find('./ObjectiveFunction/Objective').text
        assert self.objective == 'SC', 'Invalid objective function'

        # # reading constraints
        self.constrs = []
        for c in xml.findall('./Constraints/*/'):
            if c.tag[:2] == 'CA':
                self.constrs.append(ConstrCapacity(c.tag, c.attrib))
            elif c.tag[:2] == 'GA':
                self.constrs.append(ConstrGame(c.tag, c.attrib))
            elif c.tag[:2] == 'BR':
                self.constrs.append(ConstrBreak(c.tag, c.attrib))
            elif c.tag[:2] == 'FA':
                self.constrs.append(ConstrFairness(c.tag, c.attrib))
            elif c.tag[:2] == 'SE':
                self.constrs.append(ConstrSeparation(c.tag, c.attrib))

    def __pre_process(self):
        if self.preprocessed:
            raise Exception('Instance was already preprocessed.')

        self.preprocessed = True
