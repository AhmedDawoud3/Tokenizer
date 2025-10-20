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
