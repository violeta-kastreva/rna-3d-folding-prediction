import string
from functools import cached_property


class TokenLibrary:
    @cached_property
    def rna_tokens(self) -> list[str]:
        return ["A", "C", "G", "U"]

    @cached_property
    def missing_residue_token(self):
        return "-"

    @cached_property
    def unknown_residue_token(self):
        return "?"

    @cached_property
    def pad_token(self) -> str:
        return "_"

    @cached_property
    def all_tokens(self) -> list[str]:
        return (
                self.rna_tokens +
                [self.missing_residue_token, self.unknown_residue_token, self.pad_token] +
                [t for t in string.ascii_uppercase if t not in self.rna_tokens]
        )

    @cached_property
    def map_token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.all_tokens)}

    @cached_property
    def rna_token_ids(self) -> list[int]:
        return [self.map_token_to_id[token] for token in self.rna_tokens]

    @cached_property
    def missing_residue_token_id(self) -> int:
        return self.map_token_to_id[self.missing_residue_token]

    @cached_property
    def unknown_residue_token_id(self) -> int:
        return self.map_token_to_id[self.unknown_residue_token]

    @cached_property
    def pad_token_id(self) -> int:
        return self.map_token_to_id[self.pad_token]
