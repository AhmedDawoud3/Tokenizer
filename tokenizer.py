from collections import Counter
from typing import List

from tqdm import tqdm


class Tokenizer:
    def __init__(self, merges: List[tuple[tuple[int, int], int]] = None):  # type: ignore
        if merges is not None:
            if not all(
                isinstance(pair, tuple) and len(pair) == 2 and isinstance(id, int)
                for pair, id in merges
            ):
                raise ValueError("Merges must be a list of ((int, int), int) tuples.")
            for i, merge in enumerate(merges):
                if merge[1] != 256 + i:
                    raise ValueError(
                        f"Merges must have consecutive IDs starting from 256. "
                        f"Found {merge[1]} at position {i}, expected {256 + i}."
                    )

        self.merges = merges if merges is not None else []
        self.vocab_size = 256 + len(self.merges)
        self.update_tokentobytes()

    def train_from_file(
        self, file_path: str, vocab_size=None, num_merges=None, verbose=False
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.train(text, vocab_size, num_merges, verbose)

    def train(
        self, text: str | List[int], vocab_size=None, num_merges=None, verbose=False
    ):
        if vocab_size is None:
            if num_merges is None:
                raise ValueError("Either vocab_size or num_merges must be provided.")
            vocab_size = self.vocab_size + num_merges

        if isinstance(text, str):
            text = self.text_to_ids(text)

        original_len = len(text)

        progress = tqdm(
            range(vocab_size - self.vocab_size),
            disable=not verbose,
            unit="merges",
            desc="Training BPE",
        )
        for _ in progress:
            stats = self.stats(text)
            replace = max(stats.items(), key=lambda x: x[1])
            pair, count = replace
            new_id = 256 + len(self.merges)
            progress.set_postfix(
                {
                    "pair": repr(chr(pair[0]) + chr(pair[1])),
                    "count": count,
                    "new_id": new_id,
                }
            )
            new_text = self.merge(text, pair, new_id)
            self.merges.append((pair, new_id))
            text = new_text

        if verbose:
            print(
                f"Original length: {original_len}, New length: {len(text)},"
                " Compression: {original_len / len(text):.2f}x"
            )

        self.vocab_size = vocab_size
        self.update_tokentobytes()

        return text

    def update_tokentobytes(self):
        self.tokentobytes = {token: bytes([token]) for token in range(256)}
        for pair, _id in self.merges:
            self.tokentobytes[_id] = (
                self.tokentobytes[pair[0]] + self.tokentobytes[pair[1]]
            )

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for pair, new_id in self.merges:
                f.write(f"{new_id}\t{pair[0]}\t{pair[1]}\n")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        merges = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                new_id, pair_0, pair_1 = line.strip().split("\t")
                merges.append(((int(pair_0), int(pair_1)), int(new_id)))
        return cls(merges)

    def encode(self, text: str) -> List[int]:
        ids = self.text_to_ids(text)
        for pair, new_id in self.merges:
            ids = self.merge(ids, pair, new_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.ids_to_text(ids)

    def merge(self, ids: List[int], pair: tuple[int, int], new_id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids) - 1:
            if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        if i < len(ids):
            new_ids.append(ids[i])

        return new_ids

    def ids_to_text(self, ids: List[int]) -> str:
        return b"".join(self.tokentobytes[id] for id in ids).decode(
            "utf-8", errors="replace"
        )

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def tokens(self) -> List[str]:
        return [t.decode("utf-8", errors="ignore") for t in self.tokentobytes.values()]

    @classmethod
    def text_to_ids(cls, text: str) -> List[int]:
        return list(map(int, text.encode("utf-8")))

    @classmethod
    def stats(cls, ids: List[int]) -> Counter[tuple[int, int]]:
        return Counter(zip(ids, ids[1:]))
