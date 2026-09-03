from enum import IntEnum


class Action(IntEnum):
    WAIT = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


ACTION_DELTAS = {
    Action.WAIT: (0, 0),
    Action.NORTH: (0, -1),
    Action.SOUTH: (0, 1),
    Action.EAST: (1, 0),
    Action.WEST: (-1, 0),
}
