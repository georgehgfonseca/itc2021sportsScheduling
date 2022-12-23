from enum import Enum


class ConstraintCategory(Enum):
    UNKNOWN = 0
    CAPACITY = 1
    GAME = 2
    BREAK = 3
    FAIRNESS = 4
    SEPARATION = 5


class Constraint:
    __slots__ = ['tag', 'data', 'penalty', 'slots', 'teams', 'type']

    category = ConstraintCategory.UNKNOWN

    def __init__(self, tag: str, data: dict):
        self.tag = tag
        self.data = data

        self.penalty = int(self.data['penalty'])
        self.slots = sorted([int(s) for s in self.data['slots'].split(';')]) if 'slots' in data else []
        self.teams = sorted([int(t) for t in self.data['teams'].split(';')]) if 'teams' in data else []
        self.type = self.data['type']


class ConstrCapacity(Constraint):
    category = ConstraintCategory.CAPACITY

    def __init__(self, tag: str, data: dict):
        super().__init__(tag, data)

        self.teams1 = [int(t) for t in self.data['teams1'].split(';')] if 'teams1' in data else []
        self.teams2 = [int(t) for t in self.data['teams2'].split(';')] if 'teams2' in data else []

        self.intp = int(data['intp']) if 'intp' in data else None
        self.max = int(data['max']) if 'max' in data else None
        self.mode = data['mode'] if 'mode' in data else None
        self.mode1 = data['mode1'] if 'mode1' in data else None
        self.mode2 = data['mode2'] if 'mode2' in data else None


class ConstrGame(Constraint):
    category = ConstraintCategory.GAME

    def __init__(self, tag: str, data: dict):
        super().__init__(tag, data)

        self.min = int(data['min']) if 'min' in data else None
        self.max = int(data['max']) if 'max' in data else None

        self.meetings = [m.split(",") for m in data["meetings"].split(";") if m != ""] if 'meetings' in data else None
        for i in range(len(self.meetings)):
            self.meetings[i] = (int(self.meetings[i][0]), int(self.meetings[i][1]))


class ConstrBreak(Constraint):
    category = ConstraintCategory.BREAK

    def __init__(self, tag: str, data: dict):
        super().__init__(tag, data)

        self.intp = int(data['intp']) if 'intp' in data else None
        self.mode2 = data['mode2'] if 'mode2' in data else None

class ConstrFairness(Constraint):
    category = ConstraintCategory.FAIRNESS

    def __init__(self, tag: str, data: dict):
        super().__init__(tag, data)

        self.intp = int(data['intp'])
        self.mode = data['mode']


class ConstrSeparation(Constraint):
    category = ConstraintCategory.CAPACITY

    def __init__(self, tag: str, data: dict):
        super().__init__(tag, data)

        self.min = int(data['min'])
        self.mode1 = data['mode1']
