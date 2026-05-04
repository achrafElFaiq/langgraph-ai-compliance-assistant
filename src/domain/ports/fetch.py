from abc import ABC, abstractmethod

from src.domain.models.models import FetchResult


class RegulationFetcher(ABC):

    @abstractmethod
    def fetch(self, regulation: str)-> FetchResult:
        pass