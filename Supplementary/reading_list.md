# TRI AI Saturdays Cohort 10 — Additional Reading List

Organized by session. Each resource includes:
- Title
- Author / Organization
- Year
- Type (Paper / Blog / Video / Course / Tool / Docs)
- URL (if freely available)
- Brief description
- Skill gap it addresses

Skill gap tags:
- `[#python]` — assumes no or limited Python
- `[#ml-basics]` — assumes no or limited ML background
- `[#nlp]` — assumes some ML, targets NLP specifically
- `[#advanced]` — for students who want to go deeper
- `[#tools]` — practical tool usage, not theory

---

## Session 1: Intro to Language Models & BPE Tokenization

### Beginner-Friendly Resources

1. **Hugging Face NLP Course — Chapter 2: Tokenizers**
   - Hugging Face
   - 2023
   - Type: Course / Docs
   - URL: https://huggingface.co/learn/nlp-course/chapter2/4
   - Description: Practical, code-first introduction to tokenization using the Hugging Face `tokenizers` library. Covers BPE, WordPiece, and SentencePiece with interactive examples.
   - Skill gap: `[#python]` `[#tools]`

2. **The Illustrated Word2vec (Tokenization Section)**
   - Jay Alammar
   - 2019
   - Type: Blog
   - URL: https://jalammar.github.io/illustrated-word2vec/
   - Description: Visual introduction to how words become numbers. The first section covers tokenization concepts in an approachable, diagram-heavy format.
   - Skill gap: `[#ml-basics]`

3. **What is Tokenization? (YouTube)**
   - Google Cloud Tech
   - 2023
   - Type: Video
   - URL: https://www.youtube.com/watch?v=CXraLK8pUIw
   - Description: Short, accessible 5-minute explanation of tokenization and why it matters for LLMs. Good warm-up before the session.
   - Skill gap: `[#ml-basics]`

### Core Readings

4. **A New Algorithm for Data Compression**
   - Philip Gage
   - 1994
   - Type: Paper
   - URL: https://www.derczynski.com/papers/archive/BPE%20Gage%201994.pdf
   - Description: The original paper introducing Byte-Pair Encoding for data compression. Short and readable. Shows the algorithm was not originally designed for NLP.
   - Skill gap: `[#advanced]`

5. **Language Models are Unsupervised Multitask Learners (GPT-2)**
   - Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever — OpenAI
   - 2019
   - Type: Paper
   - URL: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
   - Description: The GPT-2 paper that popularized byte-level BPE tokenization. Section 2.1 (Tokenization) explains why they chose BPE over alternatives.
   - Skill gap: `[#nlp]`

6. **The Illustrated GPT-2**
   - Jay Alammar
   - 2019
   - Type: Blog
   - URL: https://jalammar.github.io/illustrated-gpt2/
   - Description: Visual walkthrough of GPT-2's architecture including tokenization, embeddings, and transformer blocks. Excellent companion to the paper.
   - Skill gap: `[#ml-basics]` `[#nlp]`

7. **HuggingFace Tokenizers Documentation**
   - Hugging Face
   - 2023
   - Type: Docs
   - URL: https://huggingface.co/docs/tokenizers/index
   - Description: Official documentation for the `tokenizers` library used in the lab. Includes API reference, performance benchmarks, and tutorials for training custom tokenizers.
   - Skill gap: `[#tools]`

8. **Neural Machine Translation of Rare Words with Subword Units**
   - Rico Sennrich, Barry Haddow, Alexandra Birch
   - 2016
   - Type: Paper
   - URL: https://arxiv.org/abs/1508.07909
   - Description: The paper that introduced BPE to NLP. Explains why subword tokenization solves the out-of-vocabulary problem for machine translation.
   - Skill gap: `[#nlp]` `[#advanced]`

---

## Session 2: Static vs Contextual Embeddings

### Beginner-Friendly Resources

9. **Word Embeddings: A Visual Explanation (YouTube)**
   - StatQuest with Josh Starmer
   - 2021
   - Type: Video
   - URL: https://www.youtube.com/watch?v=ERibwQs9hD0
   - Description: Josh Starmer's clear, animated explanation of word embeddings. Covers why we need them and how Word2Vec works. No math beyond basic addition.
   - Skill gap: `[#ml-basics]`

10. **Google's Embedding Projector**
    - Google / TensorFlow
    - 2016
    - Type: Tool
    - URL: https://projector.tensorflow.org/
    - Description: Interactive web tool that lets you visualize word embeddings in 3D. Upload your own vectors or explore pre-loaded Word2Vec / GloVe embeddings. Great for building intuition.
    - Skill gap: `[#tools]`

11. **Word2Vec from Scratch with Python (YouTube)**
    - Aladdin Persson
    - 2021
    - Type: Video
    - URL: https://www.youtube.com/watch?v=Ts9SJf3JcPk
    - Description: Step-by-step implementation of Word2Vec in Python using NumPy. Good for students who want to understand the mechanics without a high-level library.
    - Skill gap: `[#python]` `[#ml-basics]`

### Core Readings

12. **Efficient Estimation of Word Representations in Vector Space**
    - Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean — Google
    - 2013
    - Type: Paper
    - URL: https://arxiv.org/abs/1301.3781
    - Description: The original Word2Vec paper. Introduces CBOW and Skip-gram architectures. Short (9 pages) and readable. The foundation of static embeddings.
    - Skill gap: `[#nlp]`

13. **Distributed Representations of Words and Phrases and their Compositionality**
    - Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean — Google
    - 2013
    - Type: Paper
    - URL: https://arxiv.org/abs/1310.4546
    - Description: Extends Word2Vec with negative sampling and subsampling of frequent words. Covers the training tricks that make Word2Vec practical at scale.
    - Skill gap: `[#nlp]` `[#advanced]`

14. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
    - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova — Google
    - 2018
    - Type: Paper
    - URL: https://arxiv.org/abs/1810.04805
    - Description: The BERT paper that revolutionized NLP with contextual embeddings. Introduces masked language modeling and next-sentence prediction pre-training.
    - Skill gap: `[#nlp]` `[#advanced]`

15. **The Illustrated BERT**
    - Jay Alammar
    - 2018
    - Type: Blog
    - URL: https://jalammar.github.io/illustrated-bert/
    - Description: Visual guide to BERT's architecture, embeddings, fine-tuning, and how context-aware representations differ from Word2Vec.
    - Skill gap: `[#ml-basics]` `[#nlp]`

16. **The Illustrated Word2vec**
    - Jay Alammar
    - 2019
    - Type: Blog
    - URL: https://jalammar.github.io/illustrated-word2vec/
    - Description: Companion visual guide to Word2Vec. Covers skip-gram, CBOW, negative sampling, and the intuition behind vector arithmetic.
    - Skill gap: `[#ml-basics]`

17. **Stanford CS224n: NLP with Deep Learning — Lecture Notes**
    - Stanford University / Chris Manning
    - 2023
    - Type: Course
    - URL: https://web.stanford.edu/class/cs224n/
    - Description: Full lecture notes from the Stanford NLP course. Covers word vectors, RNNs, Transformers, and BERT. Lecture 2 focuses on Word2Vec; Lecture 10 on BERT.
    - Skill gap: `[#nlp]` `[#advanced]`

18. **Gemma: A Family of Lightweight Open Models from Google DeepMind**
    - Google DeepMind
    - 2024
    - Type: Paper / Blog
    - URL: https://ai.google.dev/gemma
    - Description: The Gemma model documentation. Includes details on tokenization (256k vocab), embedding dimensions, and architecture. Directly relevant to the models used in this cohort.
    - Skill gap: `[#nlp]`

19. **Representation Learning: A Review and New Perspectives**
    - Yoshua Bengio, Aaron Courville, Pascal Vincent
    - 2013
    - Type: Paper
    - URL: https://arxiv.org/abs/1206.5538
    - Description: Broad survey of representation learning, including embeddings. Heavy on theory. Best for students who want to understand the "why" behind embeddings at a deep level.
    - Skill gap: `[#advanced]`

20. **Contextual Embeddings for Cross-Lingual Transfer in African Languages**
    - Various (Masakhane NLP)
    - 2020–2024
    - Type: Blog / Research (multiple papers)
    - URL: https://www.masakhane.io/
    - Description: Collection of research from the Masakhane NLP community applying contextual embeddings to African languages. Directly relevant to the cohort's mission.
    - Skill gap: `[#nlp]` `[#advanced]`

---

## Cross-Session & Reference

21. **The Annotated Transformer**
    - Harvard NLP / Alexander Rush
    - 2018
    - Type: Blog / Course
    - URL: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - Description: Line-by-line implementation and explanation of the Transformer architecture in PyTorch. Essential for understanding how contextual embeddings are generated.
    - Skill gap: `[#nlp]` `[#advanced]`

22. **Speech and Language Processing (Chapter 6: Vector Semantics)**
    - Dan Jurafsky, James H. Martin
    - 2023 (3rd ed. draft)
    - Type: Course / Textbook
    - URL: https://web.stanford.edu/~jurafsky/slp3/
    - Description: Chapter 6 covers vector semantics and embeddings comprehensively. Free online draft. Excellent reference for both static and contextual embeddings.
    - Skill gap: `[#ml-basics]` `[#nlp]`

23. **Python for Everybody (Free Course)**
    - Charles Severance / University of Michigan
    - 2019
    - Type: Course
    - URL: https://www.py4e.com/
    - Description: Free introductory Python course for absolute beginners. Suggested prerequisite for students who have no programming background before attempting the lab code.
    - Skill gap: `[#python]`

24. **Practical Data Science with Python (Kaggle)**
    - Kaggle
    - 2023
    - Type: Course
    - URL: https://www.kaggle.com/learn
    - Description: Free micro-courses on Python, Pandas, NumPy, and ML basics. Students with no NumPy/Pandas experience (45% of cohort) should start here before the sessions.
    - Skill gap: `[#python]` `[#ml-basics]` `[#tools]`
