from typing import Set


class IdSet:

    def __init__(self, prefix=''):
        self.counter: int = 0
        self.id: Set[str] = set()
        self.prefix = prefix

    def add(self, element: str) -> str:
        element = element.strip().upper()
        self.id.add(element)
        return element
