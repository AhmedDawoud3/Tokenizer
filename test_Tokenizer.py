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


class TestTokenizerArabic:  # type: ignore
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


class TestTokenizerArabicRegex:
    """Test Arabic text with tashkil using regex-based tokenization."""

    def test_regex_split_arabic_with_tashkil(self):
        """Test that regex correctly splits Arabic text with tashkil."""
        tokenizer = Tokenizer()
        text = "مَرْحَبًا بِكَ فِي الْعَالَمِ"
        chunks = tokenizer.regex_split(text)
        assert len(chunks) > 0
        # Verify that rejoining chunks reconstructs the original text
        assert "".join(chunks) == text

    def test_encode_decode_arabic_regex_with_tashkil(self):
        """Test encoding and decoding Arabic text with tashkil using regex."""
        tokenizer = Tokenizer()
        arabic_text = "مَرْحَبًا بِكَ فِي الْعَالَمِ"
        chunks = tokenizer.regex_split(arabic_text)
        ids = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in ids for item in sublist])  # type: ignore
        assert decoded == arabic_text

    def test_train_arabic_regex_with_tashkil(self):
        """Test training on Arabic text with tashkil using regex."""
        text = "مَرْحَبًا مَرْحَبًا بِكَ بِكَ فِي فِي الْعَالَمِ الْعَالَمِ"
        tokenizer = Tokenizer()
        tokenizer.train(text, num_merges=15, verbose=False)
        assert tokenizer.vocab_size == 271
        assert len(tokenizer.merges) == 15

    def test_arabic_tashkil_compression_regex(self):
        """Test that Arabic with tashkil gets compressed during training with regex."""
        text = "اللَّهُ اللَّهُ اللَّهُ الرَّحْمَٰنِ الرَّحْمَٰنِ الرَّحِيمِ الرَّحِيمِ"
        tokenizer = Tokenizer()
        chunks = tokenizer.regex_split(text)
        original_ids = tokenizer.encode(chunks)
        original_len = sum(len(seq) for seq in original_ids)  # type: ignore

        tokenizer.train(text, num_merges=25, verbose=False)
        chunks = tokenizer.regex_split(text)
        compressed_ids = tokenizer.encode(chunks)
        compressed_len = sum(len(seq) for seq in compressed_ids)  # type: ignore

        assert compressed_len < original_len
        decoded = tokenizer.decode([item for sublist in compressed_ids for item in sublist])  # type: ignore
        assert decoded == text

    def test_mixed_arabic_english_regex_with_tashkil(self):
        """Test encoding mixed Arabic (with tashkil) and English text with regex."""
        tokenizer = Tokenizer()
        mixed_text = "Hello مَرْحَبًا World عَالَم"
        chunks = tokenizer.regex_split(mixed_text)
        ids = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in ids for item in sublist])  # type: ignore
        assert decoded == mixed_text

    def test_quran_verse_regex_with_full_tashkil(self):
        """Test complete Quranic verse with full tashkil using regex."""
        tokenizer = Tokenizer()
        quran_verse = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        chunks = tokenizer.regex_split(quran_verse)
        ids = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in ids for item in sublist])  # type: ignore
        assert decoded == quran_verse
        assert len(chunks) > 0

    def test_arabic_tashkil_repeated_patterns_regex(self):
        """Test that repeated Arabic patterns with tashkil get merged efficiently with regex."""
        text = "كَتَبَ كَتَبَ كَتَبَ قَرَأَ قَرَأَ قَرَأَ"
        tokenizer = Tokenizer()
        tokenizer.train(text, num_merges=20, verbose=False)

        chunks = tokenizer.regex_split(text)
        encoded = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in encoded for item in sublist])  # type: ignore
        assert decoded == text
        assert tokenizer.vocab_size == 276

    def test_arabic_sentence_boundaries_regex(self):
        """Test that regex properly handles Arabic sentence boundaries with tashkil."""
        tokenizer = Tokenizer()
        text = "السَّلَامُ عَلَيْكُمْ. كَيْفَ حَالُكَ؟"
        chunks = tokenizer.regex_split(text)
        # Should split on punctuation
        assert len(chunks) > 2
        ids = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in ids for item in sublist])  # type: ignore
        assert decoded == text

    def test_arabic_numbers_with_tashkil_regex(self):
        """Test Arabic text with numbers and tashkil using regex."""
        tokenizer = Tokenizer()
        text = "لَدَيْنَا 123 كِتَابًا"
        chunks = tokenizer.regex_split(text)
        ids = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in ids for item in sublist])  # type: ignore
        assert decoded == text

    def test_long_arabic_text_with_tashkil_regex(self):
        """Test longer Arabic text with tashkil for realistic compression."""
        text = """قَالَ اللَّهُ تَعَالَى: بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ. 
        الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ الرَّحْمَٰنِ الرَّحِيمِ مَالِكِ يَوْمِ الدِّينِ.
        إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ. اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ."""

        tokenizer = Tokenizer()
        chunks_before = tokenizer.regex_split(text)
        ids_before = tokenizer.encode(chunks_before)
        len_before = sum(len(seq) for seq in ids_before)  # type: ignore

        tokenizer.train(text, num_merges=50, verbose=False)

        chunks_after = tokenizer.regex_split(text)
        ids_after = tokenizer.encode(chunks_after)
        len_after = sum(len(seq) for seq in ids_after)  # type: ignore

        decoded = tokenizer.decode([item for sublist in ids_after for item in sublist])  # type: ignore

        assert decoded == text
        assert len_after < len_before
        assert tokenizer.vocab_size == 306

    def test_arabic_with_multiple_tashkil_regex(self):
        """Test Arabic words with multiple consecutive tashkil marks."""
        tokenizer = Tokenizer()
        text = "مُحَمَّدٌ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"
        chunks = tokenizer.regex_split(text)
        ids = tokenizer.encode(chunks)
        decoded = tokenizer.decode([item for sublist in ids for item in sublist])  # type: ignore
        assert decoded == text

    def test_regex_preserves_tashkil_order(self):
        """Test that regex splitting preserves the order of tashkil marks."""
        tokenizer = Tokenizer()
        # Text with various tashkil: fatha, kasra, damma, sukun, shadda, tanween
        text = "مَكْتُوبٌ بِالْحَرَكَاتِ الْمُخْتَلِفَةِ"
        chunks = tokenizer.regex_split(text)
        rejoined = "".join(chunks)
        assert rejoined == text

    def test_train_from_chunks_arabic_with_tashkil(self):
        """Test training directly from regex-split chunks."""
        text = "الْكِتَابُ الْكِتَابُ الْقَلَمُ الْقَلَمُ الْمَدْرَسَةُ الْمَدْرَسَةُ"
        tokenizer = Tokenizer()

        result = tokenizer.train(text, num_merges=20, verbose=False)

        # Verify the result is a list of lists (chunks)
        assert isinstance(result, list)
        assert all(isinstance(chunk, list) for chunk in result)

        # Decode and verify
        decoded = tokenizer.decode([item for sublist in result for item in sublist])  # type: ignore
        assert decoded == text
