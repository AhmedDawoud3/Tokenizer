import pytest
from tokenizer import Tokenizer


class TestTokenizerInit:
    def test_init_with_no_merges(self):
        """Test initialization with no merges (default)."""
        tokenizer = Tokenizer()
        assert tokenizer.merges == []
        assert tokenizer.vocab_size == 256
        assert len(tokenizer.tokentobytes) == 256

    def test_init_with_valid_merges(self):
        """Test initialization with valid merges."""
        merges = [((97, 98), 256), ((99, 100), 257)]
        tokenizer = Tokenizer(merges)
        assert tokenizer.merges == merges
        assert tokenizer.vocab_size == 258
        assert len(tokenizer.tokentobytes) == 258

    def test_init_with_single_merge(self):
        """Test initialization with a single merge."""
        merges = [((65, 66), 256)]
        tokenizer = Tokenizer(merges)
        assert tokenizer.merges == merges
        assert tokenizer.vocab_size == 257

    def test_init_with_invalid_merge_format(self):
        """Test that invalid merge format raises ValueError."""
        invalid_merges = [((97, 98), "256")]  # ID should be int, not string
        with pytest.raises(ValueError, match="Merges must be a list of"):
            Tokenizer(invalid_merges)  # type: ignore

    def test_init_with_invalid_pair_length(self):
        """Test that invalid pair length raises ValueError."""
        invalid_merges = [((97,), 256)]  # Pair should have 2 elements
        with pytest.raises(ValueError, match="Merges must be a list of"):
            Tokenizer(invalid_merges)  # type: ignore

    def test_init_with_non_tuple_pair(self):
        """Test that non-tuple pair raises ValueError."""
        invalid_merges = [([97, 98], 256)]  # Pair should be tuple, not list
        with pytest.raises(ValueError, match="Merges must be a list of"):
            Tokenizer(invalid_merges)  # type: ignore

    def test_init_with_non_consecutive_ids(self):
        """Test that non-consecutive IDs raise ValueError."""
        invalid_merges = [((97, 98), 256), ((99, 100), 258)]  # Should be 257, not 258
        with pytest.raises(ValueError, match="Merges must have consecutive IDs"):
            Tokenizer(invalid_merges)  # type: ignore

    def test_init_with_wrong_starting_id(self):
        """Test that wrong starting ID raises ValueError."""
        invalid_merges = [((97, 98), 257)]  # First merge should start at 256
        with pytest.raises(ValueError, match="Merges must have consecutive IDs"):
            Tokenizer(invalid_merges)  # type: ignore

    def test_init_with_multiple_consecutive_merges(self):
        """Test initialization with multiple consecutive merges."""
        merges = [
            ((97, 98), 256),
            ((99, 100), 257),
            ((101, 102), 258),
            ((103, 104), 259),
        ]
        tokenizer = Tokenizer(merges)
        assert tokenizer.vocab_size == 260
        assert len(tokenizer.merges) == 4

    def test_init_tokentobytes_mapping(self):
        """Test that tokentobytes is correctly initialized."""
        merges = [((97, 98), 256)]
        tokenizer = Tokenizer(merges)
        assert tokenizer.tokentobytes[97] == b"a"
        assert tokenizer.tokentobytes[98] == b"b"
        assert tokenizer.tokentobytes[256] == b"ab"


class TestTokenizerArabic: # type: ignore
    def test_encode_decode_arabic_with_tashkil(self):
        """Test encoding and decoding Arabic text with tashkil (diacritics)."""
        tokenizer = Tokenizer()
        arabic_text = "مَرْحَبًا بِكَ فِي الْعَالَمِ"
        ids = tokenizer.encode(arabic_text)
        decoded = tokenizer.decode(ids)
        assert decoded == arabic_text
        assert all(isinstance(id, int) for id in ids)

    def test_train_arabic_with_tashkil(self):
        """Test training on Arabic text with tashkil."""
        text = "مَرْحَبًا مَرْحَبًا بِكَ بِكَ فِي فِي الْعَالَمِ الْعَالَمِ"
        tokenizer = Tokenizer()
        tokenizer.train(text, num_merges=10, verbose=False)
        assert tokenizer.vocab_size == 266
        assert len(tokenizer.merges) == 10

    def test_arabic_tashkil_compression(self):
        """Test that Arabic with tashkil gets compressed during training."""
        text = "اللَّهُ اللَّهُ اللَّهُ الرَّحْمَٰنِ الرَّحْمَٰنِ الرَّحِيمِ الرَّحِيمِ"
        tokenizer = Tokenizer()
        original_ids = tokenizer.encode(text)
        original_len = len(original_ids)

        tokenizer.train(text, num_merges=20, verbose=False)
        compressed_ids = tokenizer.encode(text)

        assert len(compressed_ids) < original_len
        assert tokenizer.decode(compressed_ids) == text

    def test_mixed_arabic_english_with_tashkil(self):
        """Test encoding mixed Arabic (with tashkil) and English text."""
        tokenizer = Tokenizer()
        mixed_text = "Hello مَرْحَبًا World عَالَم"
        ids = tokenizer.encode(mixed_text)
        decoded = tokenizer.decode(ids)
        assert decoded == mixed_text

    def test_arabic_tashkil_tokentobytes(self):
        """Test tokentobytes mapping for Arabic with tashkil after merges."""
        merges = [((216, 167), 256), ((217, 132), 257)]  # Common Arabic byte pairs
        tokenizer = Tokenizer(merges)
        assert 256 in tokenizer.tokentobytes
        assert 257 in tokenizer.tokentobytes
        assert len(tokenizer.tokentobytes[256]) == 2
        assert len(tokenizer.tokentobytes[257]) == 2

    def test_quran_verse_with_full_tashkil(self):
        """Test complete Quranic verse with full tashkil."""
        tokenizer = Tokenizer()
        quran_verse = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        ids = tokenizer.encode(quran_verse)
        decoded = tokenizer.decode(ids)
        assert decoded == quran_verse
        assert len(ids) > 0

    def test_arabic_tashkil_repeated_patterns(self):
        """Test that repeated Arabic patterns with tashkil get merged efficiently."""
        text = "كَتَبَ كَتَبَ كَتَبَ قَرَأَ قَرَأَ قَرَأَ"
        tokenizer = Tokenizer()
        tokenizer.train(text, num_merges=15, verbose=False)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text
        assert tokenizer.vocab_size == 271
