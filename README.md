# 🔤 Tokenizer - BPE Tokenizer with Arabic Support

A powerful Byte Pair Encoding (BPE) tokenizer implementation with full **Arabic language support**, including Arabic diacritics (تشكيل).
This tokenizer is built from the ground up for clarity and extensibility.

---

## ✨ Features

- **🌍 Arabic Language Support**: Full support for Arabic text, including Arabic diacritics (Tashkil/تشكيل)
- **📝 Flexible Regex Patterns**: Support for both GPT-2 and GPT-4 style tokenization patterns, **modified to handle Arabic diacritics** (combining marks)
- **💾 Save/Load Models**: Persist and reload trained tokenizers
- **🎯 BPE Training**: Train custom tokenizers on your own text data
- **📊 Progress Tracking**: Built-in progress bars for training visualization
- **🔄 Encode/Decode**: Convert between text and token IDs seamlessly
- **📈 Compression Stats**: View tokenization compression ratios

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

### Requirements

- `regex` - Advanced regex support for Unicode patterns
- `tqdm` - Progress bar visualization
- `colorama` - Terminal color support

---

## 📖 Usage

### Basic Example

```python
from tokenizer import Tokenizer

# Initialize a tokenizer
tokenizer = Tokenizer()

# Train on text
text = "مرحباً بك في عالم معالجة اللغة الطبيعية!"
tokenizer.train(text, vocab_size=512, verbose=True)

# Encode text to token IDs
ids = tokenizer.encode("مرحباً")
print(f"Token IDs: {ids}") # Token IDs: [272]

# Decode back to text
decoded = tokenizer.decode(ids)
print(f"Decoded text: {decoded}") #Decoded text: مرحباً

# Save the trained model
tokenizer.save("arabic_tokenizer.model")

# Load a saved model
loaded_tokenizer = Tokenizer.load("arabic_tokenizer.model")
```

### Training from File

```python
tokenizer = Tokenizer()
tokenizer.train_from_file("corpus.txt", vocab_size=1000, verbose=True)
```

### Custom Regex Patterns

```python
from tokenizer import Tokenizer, GPT4_SPLIT_PATTERN, GPT2_SPLIT_PATTERN

# Use GPT-4 pattern (default)
tokenizer = Tokenizer(regex_pattern=GPT4_SPLIT_PATTERN)

# Or use GPT-2 pattern
tokenizer = Tokenizer(regex_pattern=GPT2_SPLIT_PATTERN)
```

---

## 🎨 What's Implemented

- [x] Core BPE tokenization algorithm
- [x] Arabic language support (normal text)
- [x] Arabic diacritics (تشكيل) support
- [x] Regex-based text splitting (GPT-2 & GPT-4 patterns)
- [x] Save/Load tokenizer models
- [x] Training progress visualization
- [x] Encode/Decode functionality

- [ ] **Colored Terminal Output**: Implement a feature to display encoded tokens in the terminal with distinct colors for better visual analysis.

- [ ] **Performance Optimization**: Speed up the encoding process for large texts
  - Implement chunked encoding for large texts !?
  - Parallel processing for multiple sequences !?
  - Memory optimization for large vocabularies !?

- [ ] **Special Tokens**: Add support for special tokens like `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, and `[MASK]`.

---

## 🤝 Contributing

**We warmly welcome contributions from the community!**

Whether you're fixing bugs, adding features, improving documentation, or enhancing Arabic language support, your contributions are valuable.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Set Up pre-commit**: This project uses black and isort for code formatting. Install and set up the pre-commit hooks to ensure your code matches the project's style.
```
pip install pre-commit
pre-commit install
```
4. **Make your changes**
5. **Add tests** for new functionality
6. **Commit your changes** The pre-commit hooks will automatically format your code when you commit.
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Areas We'd Love Help With

- 🎨 Implementing colored terminal output for tokens
- ⚡ Performance optimization and chunked encoding
- 🌍 Improving Arabic language support
- 📚 Documentation improvements
- 🧪 Additional test cases
- 🐛 Bug fixes and code quality improvements
- 💡 New features and ideas

### Contribution Guidelines

- Write clear, descriptive commit messages
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Be respectful and constructive in discussions

**Don't hesitate to open an issue if you have questions or suggestions!**

---

## 📝 Technical Details

### Supported Regex Patterns

#### GPT-4 Pattern (Default)

Better for Unicode and Arabic support. **Modified to include `\p{Mn}*` (combining marks) for proper Arabic tashkil handling**:

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+(?:\p{L}\p{Mn}*)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

#### GPT-2 Pattern

**Enhanced with `\p{Mn}*` for Arabic diacritics support**:

```
's|'t|'re|'ve|'m|'ll|'d| ?(?:\p{L}\p{Mn}*)+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

> **Note**: The `\p{Mn}*` pattern matches zero or more combining marks (non-spacing marks), which is essential for correctly handling Arabic diacritical marksز

### How BPE Works

1. **Initialize**: Start with all bytes (0-255) as base tokens
2. **Count Pairs**: Find the most frequent pair of consecutive tokens
3. **Merge**: Replace all occurrences of that pair with a new token
4. **Repeat**: Continue until desired vocabulary size is reached

---

## 🙏 Acknowledgments

- Tests auto-generated with **Claude Sonnet 4.5**
- Inspired by OpenAI's BPE implementation
- Arabic language support designed for NLP researchers and developers

---

## 📬 Contact

Have questions or suggestions? Feel free to:

- Open an issue
- Submit a pull request
- Reach out to the maintainers

**Let's build something amazing together! 🚀**
