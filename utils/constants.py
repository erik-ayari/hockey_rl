from enum import Enum

class EnvironmentType(Enum):
    REGULAR = 0
    GAME    = 1

class AgentType(Enum):
    SINGLE_AGENT    = 0
    MULTI_AGENT     = 1

class SplitActionSpace(Enum):
    NO_SPLIT    = 0
    SPLIT       = 1