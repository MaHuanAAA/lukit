from typing import Dict, Iterable, List

from ..data_providers import PROVIDER_REGISTRY, DataProvider


class DependencyResolver:
    """Resolve required data keys to a minimal provider execution plan."""

    def __init__(self, provider_registry: Dict[str, type] = None) -> None:
        self.provider_registry = provider_registry or PROVIDER_REGISTRY

    def resolve(self, required_data: Iterable[str]) -> List[DataProvider]:
        providers: List[DataProvider] = []
        for data_key in sorted(set(required_data)):
            provider_cls = self.provider_registry.get(data_key)
            if provider_cls is None:
                raise KeyError(f"No data provider registered for: {data_key}")
            providers.append(provider_cls())
        return providers

    def collect_model_requirements(self, required_data: Iterable[str]) -> List[str]:
        needs = set()
        for provider in self.resolve(required_data):
            needs.update(provider.requires_from_model)
        return sorted(needs)
