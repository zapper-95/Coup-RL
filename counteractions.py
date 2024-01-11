import abc as ABC, abstractmethod

class Counteraction(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def counteracts