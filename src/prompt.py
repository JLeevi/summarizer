from enum import Enum, auto

class PromptStyle(Enum):
    BASIC = auto()
    CHAT = auto()
    DESCRIPTIVE = auto()
    EMPTY = auto()
    DESCRIPTIVE_FORCE_LENGTH = auto()