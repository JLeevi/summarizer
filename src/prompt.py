from enum import Enum, auto

class PromptStyle(Enum):
    BASIC = auto()
    CHAT = auto()
    DESCRIPTIVE = auto()
    EMPTY = auto()
    BASIC_FORCE_LENGTH = auto()