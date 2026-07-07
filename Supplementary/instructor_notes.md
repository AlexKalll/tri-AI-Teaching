# TRI AI Saturdays Cohort 10 — Instructor Notes

## Session Schedule & Timing

### Session 1: Intro to Language Models & BPE Tokenization

| Time (min) | Segment | Description |
|------------|---------|-------------|
| 0–10 | Welcome & poll (Q1) | Introductions, cohort overview, warm-up poll |
| 10–30 | Lecture: What is a Language Model? | High-level overview, n-grams to neural LMs |
| 30–35 | Poll: Prior tokenizer experience (Q1) | Quick show of hands / digital poll |
| 35–50 | Lecture: BPE Tokenization | Walk through the 5 steps with examples |
| 50–65 | Live coding: BPE from scratch | Code the 5 functions together |
| 65–75 | Quiz (Q2–Q5) | Multiple choice during-lab questions |
| 75–90 | Lab: Tokenizer training | Hands-on with `tokenizers` library |
| 90–100 | Open-ended discussion (Q6) | Medical vs agricultural text tokenization |
| 100–110 | Project connections & wrap-up | How this connects to their projects |
| 110–115 | Self-assessment poll (Q7) | Confidence check |
| 115–120 | Q&A, next session preview | |

**Total: ~120 minutes**

### Session 2: Static vs Contextual Embeddings

| Time (min) | Segment | Description |
|------------|---------|-------------|
| 0–5 | Welcome & recap of Session 1 | Quick review of BPE |
| 5–10 | Poll: Analogies (Q1) | "king - man + woman = ?" |
| 10–30 | Lecture: Static Embeddings (Word2Vec) | Intuition, training, vector arithmetic |
| 30–35 | Demo: Embedding Projector | Interactive visualization at projector.tensorflow.org |
| 35–45 | Lecture: Contextual Embeddings (BERT) | Transformer architecture, self-attention |
| 45–50 | The "Map of Meaning" metaphor | Paper atlas vs Google Maps |
| 50–60 | Quiz (Q2–Q5) | During-lab multiple choice and poll |
| 60–80 | Lab: Comparing embeddings | Hands-on with BERT vs Word2Vec |
| 80–95 | African context discussion | Yoruba "igba", agricultural chatbots, code-switching |
| 95–110 | Open-ended (Q6) + project connection | How students will use embeddings |
| 110–115 | Poll: Session rating (Q7) | |
| 115–120 | Closing, next steps | |

**Total: ~120 minutes**

---

## Common Student Questions and Answers

### Session 1

**Q: Why can't we just use whole words as tokens?**
A: You can, and early NLP systems did. But it creates two problems: (1) you'll miss words not in your vocabulary (OOV), and (2) you lose relationships between similar words (e.g., "run" and "running" are unrelated tokens). Subword tokenization splits the difference — common words stay whole, rare words decompose into meaningful subunits.

**Q: How do I choose the vocabulary size?**
A: It's a trade-off. A larger vocabulary means fewer tokens per word (faster generation, cheaper inference) but more parameters in the embedding layer. A smaller vocabulary means more tokens per word (slower, more expensive) but fewer parameters. Guideline: start with 8,000–16,000 for small models, 32,000–50,000 for standard LMs, 100,000+ for large models. Look at the token-to-word ratio on your validation set.

**Q: What's the difference between BPE, WordPiece, and SentencePiece?**
A: BPE merges the most frequent pair greedily. WordPiece merges the pair that maximizes the likelihood of the training data (frequency + probability). SentencePiece treats the input as a raw byte stream (no pre-tokenization), uses BPE or Unigram LM, and can handle any language without knowing spaces.

**Q: Does `</w>` matter?**
A: Yes. Without it, BPE doesn't know where word boundaries are and could merge across words. For example, "the cat" could get merged into "th" + "ecat". The `</w>` marker preserves word boundary information. Some modern implementations (like SentencePiece) use a different approach (treating space as a regular character).

**Q: Can I train a tokenizer for a low-resource African language?**
A: Absolutely. That's one of the strengths of BPE. You need as little as 1–10 MB of text to train a reasonable tokenizer. The model will learn the most common subword patterns in that language, even if it's not written with spaces (e.g., many Bantu languages use agglutination).

### Session 2

**Q: What's the difference between an embedding and a vector?**
A: In practice, they're used interchangeably for word representations. Technically: a vector is the mathematical object (a list of numbers); an embedding is the mapping from a word to its vector. The embedding *layer* or *matrix* stores all the vectors. The embedding *vector* is the output for a specific word.

**Q: If BERT is so much better, why would anyone use Word2Vec?**
A: Three reasons: (1) Word2Vec is much faster to train and use (seconds vs hours), (2) Word2Vec embeddings are smaller and easier to deploy on edge devices, and (3) for tasks that don't need context (simple similarity, clustering, feature extraction for linear models), Word2Vec often performs comparably while being simpler. Always start with the simplest approach.

**Q: How are embeddings used in practice?**
A: Three main patterns:
- **As features:** Freeze the embeddings, feed them into a classifier (e.g., logistic regression on top of BERT's [CLS] token)
- **Fine-tuning:** Load pre-trained weights and continue training on your task (full model update)
- **Retrieval:** Use embeddings as search keys — convert documents to vectors, store them in a vector database, and retrieve by cosine similarity

**Q: What does "768-dimensional" mean?**
A: Each word is represented by 768 numbers. Think of it as 768 different "features" the model learned — some might correspond to grammatical properties, some to semantic categories, others to word frequency, etc. Most are not interpretable to humans. Higher dimensions (1024, 4096) can capture more nuanced relationships but require more parameters and data.

**Q: Can embeddings work for African languages that aren't in the training data?**
A: Yes, but with caveats. Multilingual BERT (mBERT) covers 104 languages including Swahili, Yoruba, Hausa, and Zulu. However, performance on African languages is generally lower because they are underrepresented in the training data. Gemma was trained on more diverse data. Fine-tuning on target language data significantly improves performance. This is an active area of research — the Masakhane community is working on improving African language NLP.

---

## Debugging Tips for Common Code Issues

### tokenizers library

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| `BpeTrainer` not found | Wrong import | `from tokenizers import trainers` then `trainers.BpeTrainer(...)` |
| Model doesn't tokenize known words | Tokenizer not trained | Call `trainer.train_from_iterator()` or `.train()` first |
| Tokens show as `[UNK]` | Word decomposed too far | Increase vocab size or add the word to the tokenizer |
| `TypeError: expected str, bytes or os.PathLike` | Passed wrong type to trainer | Ensure your corpus is `List[str]` not `List[List[str]]` |
| Training is very slow | Large corpus without normalization | Pre-tokenize and normalize first, or use `BpeTrainer(min_frequency=2)` |
| Merge count mismatch | Off-by-one error | Remember: merges = target_vocab - initial_vocab. Check vocab size before and after. |

### NumPy / Matplotlib

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'numpy'` | Package not installed | `pip install numpy matplotlib` (in correct env) |
| Plot shows in console but not in notebook | Missing `%matplotlib inline` | Add `%matplotlib inline` at top of notebook |
| `Jupyter cannot find kernel` | Wrong Python env | Install `ipykernel` in your venv, then `python -m ipykernel install --user --name=tri_env` |
| `np.load()` fails on text files | Wrong format | Use `np.loadtxt()` for text files, `np.load()` for `.npy` files |

---

## Strategies for Mixed-Skill-Level Classrooms

### Cohort Skill Profile

Based on pre-cohort survey:
- **30%** have no Python experience
- **45%** have no NumPy/Pandas experience
- **38%** have no ML experience
- **~25%** have some prior experience

### Pair Programming Suggestions

Assign pairs strategically:
- **Beginner + Intermediate:** The intermediate student explains concepts as they code together. The beginner types and asks questions. This reinforces learning for both.
- **Intermediate + Advanced:** The advanced student works on extensions while helping the intermediate with the core lab.
- **Beginner + Beginner (with TA):** Use only for the first 30 minutes, then merge pairs to ensure no pair is stuck.

Rotate pairs each session so students experience different teaching/learning dynamics.

### Extension Exercises for Advanced Students

Session 1:
- Implement BPE from scratch in pure Python (no `tokenizers` library)
- Compare tokenization across 3 languages (e.g., English, Swahili, Yoruba)
- Implement WordPiece or Unigram LM and compare merge strategies
- Analyze token-to-word ratio vs. vocabulary size for different corpora

Session 2:
- Implement Word2Vec (skip-gram + negative sampling) from scratch in NumPy
- Fine-tune BERT on a small classification task (e.g., sentiment on tweets)
- Compare embedding spaces across languages using alignment (Procrustes)
- Build a simple semantic search engine using embeddings and cosine similarity

### Extra Support for Beginners

Session 1:
- Provide a pre-commented version of the BPE code that explains each line
- Walk through the first merge step by hand on the board before coding
- Give them a prepared `tokenizer.bin` file so they can skip training and go straight to tokenizing
- Use analogies: BPE is like building LEGO — you start with the smallest bricks and combine the most common ones

Session 2:
- Start with the "Map of Meaning" metaphor before any math or code
- Use the TensorFlow Embedding Projector (no code needed) for first 10 minutes of the embeddings section
- Provide a pre-loaded dictionary of word vectors so they don't need to train from scratch
- Focus on the *what* and *why*, not the implementation details

---

## Discussion Prompts for Engagement

### Session 1

1. **"What would happen if we set vocab_size = 10 for an English corpus?"** (Follow-up: what about for a specific domain like medical text?)

2. **"If BPE didn't exist, how would you design a tokenizer?"** (Encourages creative thinking before revealing the "right" answer.)

3. **"How would you tokenize a language without spaces, like Mandarin Chinese?"** (Connects to African languages like Yoruba or Igbo.)

4. **"BPE was invented for compression in 1994. Can you see the connection between compression and language modeling?"** (Compression = predict next byte; LM = predict next token.)

### Session 2

1. **"If words were on a map, what would the 'axes' represent?"** (No right answer — encourages speculation about interpretability.)

2. **"Can you think of words in your language that change meaning depending on context?"** (Brings in African language diversity.)

3. **"What's more important for your project: speed or accuracy? Would you choose Word2Vec or BERT?"** (Forces a concrete design decision.)

4. **"How would you explain embeddings to a farmer who wants an agricultural chatbot?"** (Tests depth of understanding — can they translate technical concepts?)

---

## Connection to Career Outcomes

37% of cohort members want AI jobs. Connect each session to specific career skills:

| Skill | Where It's Used | How Session Teaches It |
|-------|----------------|----------------------|
| Tokenization | Data preprocessing at any NLP company | Session 1 lab: training tokenizers |
| Model architecture understanding | ML engineering interviews | Sessions 1–2: BPE, Word2Vec, Transformer |
| Python / NumPy | Every ML role | Lab code, test_setup.py |
| Pre-trained model usage | Applied ML roles | Session 2: loading and using pre-trained embeddings |
| Data analysis | Product/analytics roles | Analyzing token distributions, embedding spaces |
| Cross-lingual NLP | African-focused AI roles (Masakhane, Lelapa AI) | Session 2 African context discussion |

**Career tip for students:** "Build a project that tokenizes text in an African language and shares the trained tokenizer on Hugging Face. This is a concrete portfolio piece that shows you can handle multilingual NLP, which is rare and valuable."

---

## Connection to Project Domains

| Domain | % of Projects | Session 1 Relevance | Session 2 Relevance |
|--------|---------------|--------------------|--------------------|
| Education | 55% | Tokenizing curriculum text, question data | Embeddings for semantic search in learning materials, student Q&A |
| Healthcare | 25% | Tokenizing clinical notes, medical terminology | Embedding patient records, symptom-disease matching |
| Climate / Agriculture | 20% | Tokenizing farmer reports, multilingual extension docs | Embedding agricultural knowledge, cross-lingual translation |

**Domain-specific discussion questions:**

- **Education:** "If you're building a Q&A bot for students, does tokenization matter for understanding questions in local languages?"
- **Healthcare:** "Medical notes contain rare technical terms. Will a small-vocab BPE tokenizer handle 'cardiopulmonary' well?"
- **Agriculture:** "Farmer reports mix English and local languages. Should tokenization be monolingual or joint?"

---

## Quick Reference: Session Materials Checklist

### Session 1
- [ ] BPE cheat sheet (printed or PDF)
- [ ] Session notebook (`.ipynb`)
- [ ] Test corpus file(s)
- [ ] Quiz questions prepared
- [ ] Embedding Projector link (for analogy demonstration if time)

### Session 2
- [ ] Embeddings summary sheet (printed or PDF)
- [ ] Session notebook (`.ipynb`)
- [ ] Pre-trained Word2Vec / BERT model references
- [ ] TensorFlow Embedding Projector link
- [ ] Quiz questions prepared

### Both Sessions
- [ ] Reading list shared
- [ ] Setup guide reminders
- [ ] Communication channel links
- [ ] Feedback form link
