from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IsomorphismMappings:
    mapping: dict

    def __post_init__(self):
        assert self.is_bijective(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def __iter__(self):
        return self.mapping.__iter__()

    def iter_reverse_mapping(self):
        return self.reverse_mapping.items()

    def is_empty(self):
        return self.mapping == {}

    @staticmethod
    def reverse_dictionary(dictionary: dict) -> dict:
        return {value: key for (key, value) in dictionary.items()}

    @staticmethod
    def is_bijective(dictionary: dict) -> bool:
        return len(dictionary) == len(
            IsomorphismMappings.reverse_dictionary(dictionary)
        )

    @property
    def reverse_mapping(self):
        return self.reverse_dictionary(self.mapping)

    def mapping_update(
        self, new_mapping: dict, reverse: bool = False
    ) -> IsomorphismMappings:
        if self.is_bijective(new_mapping):
            if reverse:
                new_mapping = self.reverse_dictionary(new_mapping)

            final_mapping = self.mapping | new_mapping

            assert len(final_mapping) == len(self.mapping) + len(
                new_mapping
            ), "The code is trying to update an isomorphism matching overwriting some keys. Bad."

            if self.is_bijective(final_mapping):
                return IsomorphismMappings(mapping=final_mapping)
            else:
                return IsomorphismMappings(mapping={})
        else:
            return IsomorphismMappings(mapping={})

    def reverse_mapping_update(self, new_mapping: dict) -> IsomorphismMappings:
        return self.mapping_update(new_mapping, reverse=True)
