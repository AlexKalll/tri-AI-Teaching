# TRI AI Saturdays Cohort 10 — Quiz Questions

## Session 1: BPE Tokenization

---

## Session 1 - Question 1
**Type:** poll
**Time:** start
**Question:** Have you used a tokenizer before?
- [ ] No, this is my first time
- [ ] Yes, in a tutorial
- [ ] Yes, in a project
**Correct Answer:** N/A (poll — no right/wrong)
**Explanation:** This is a warm-up poll to gauge the class's prior experience with tokenization so the instructor can adjust pacing.
**Connection:** Tokenizers are the first step in every NLP pipeline. Knowing your class's baseline helps decide how much time to spend on fundamentals vs. advanced topics.

---

## Session 1 - Question 2
**Type:** multiple-choice
**Time:** during
**Question:** What does BPE stand for?
- [ ] Byte-Pair Encoding
- [ ] Binary Positional Encoding
- [ ] Back-Propagation Engine
- [ ] Bigram Pattern Extractor
**Correct Answer:** Byte-Pair Encoding
**Explanation:** BPE was originally developed for data compression by Philip Gage in 1994. It was adapted for NLP by the GPT and BERT papers as a subword tokenization method.
**Connection:** BPE is the tokenization algorithm used by GPT-2, GPT-3, LLaMA, and many other LLMs. Understanding it is key to understanding how models see text.

---

## Session 1 - Question 3
**Type:** multiple-choice
**Time:** during
**Question:** What happens in each BPE iteration?
- [ ] A random pair of symbols is merged
- [ ] The least frequent pair is removed
- [ ] The most frequent adjacent symbol pair is merged into a new token
- [ ] The vocabulary is sorted alphabetically
**Correct Answer:** The most frequent adjacent symbol pair is merged into a new token
**Explanation:** BPE greedily picks the most frequent pair of adjacent symbols across the entire corpus, merges them into a single new token, then repeats. This is the core loop of the algorithm.
**Connection:** This greedy merge strategy is what allows BPE to learn common subword units like "ing", "tion", and "pre" without any linguistic knowledge.

---

## Session 1 - Question 4
**Type:** multiple-choice
**Time:** during
**Question:** What is the purpose of the `</w>` marker in BPE?
- [ ] It marks the end of a sentence
- [ ] It indicates a punctuation mark
- [ ] It marks the end of a word
- [ ] It is used for byte-level encoding
**Correct Answer:** It marks the end of a word
**Explanation:** The `</w>` marker tells BPE where word boundaries occur. Without it, the algorithm might merge symbols across word boundaries, which would lose important information about where words start and end.
**Connection:** Word boundaries are critical for languages like Yoruba and Swahili where agglutination (combining morphemes) is common. `</w>` helps the tokenizer learn meaningful subword units.

---

## Session 1 - Question 5
**Type:** poll
**Time:** during
**Question:** If your target vocabulary size is 50 tokens and your initial character set has 20 unique characters, how many merges will BPE perform?
- [ ] 10
- [ ] 20
- [ ] 30
- [ ] 50
**Correct Answer:** 30
**Explanation:** Each merge adds exactly one new token to the vocabulary. Initial vocab = 20 characters. Target = 50. Merges needed = 50 - 20 = 30. This is a common calculation when configuring tokenizers.
**Connection:** When training a tokenizer for an African language corpus, you need to decide the target vocabulary size. This calculation helps you understand how many merge operations the algorithm will need.

---

## Session 1 - Question 6
**Type:** open-ended
**Time:** end
**Question:** How would tokenization differ for medical text (e.g., clinical notes with terms like "cardiopulmonary" and "hypertension") vs. agricultural text (e.g., farmer reports in local languages with terms like "cassava", "maize", "intercropping")?
**Correct Answer:** (Open-ended — evaluate for reasoning)
**Explanation:** Medical text: vocabulary is more standardized (Latin/Greek roots), BPE will learn prefixes/suffixes like "cardio-", "-ectomy", "hyper-". Agricultural text: may involve code-switching (local language + English), more diverse morphology, BPE needs more merges for agglutinative languages.
**Connection:** This connects directly to cohort project domains: 25% Healthcare, 20% Climate/Agriculture, 55% Education. Students should think about how tokenizer choices affect downstream model performance in their domain.

---

## Session 1 - Question 7
**Type:** poll
**Time:** end
**Question:** I can now train and apply a BPE tokenizer from scratch.
- [ ] 1 — Strongly disagree
- [ ] 2 — Disagree
- [ ] 3 — Neutral
- [ ] 4 — Agree
- [ ] 5 — Strongly agree
**Correct Answer:** N/A (poll — self-assessment)
**Explanation:** This self-assessment helps the instructor gauge whether students feel confident with the lab material and whether review or additional practice sessions are needed.
**Connection:** Tracking confidence levels across sessions helps the TRI AI team improve future cohorts.

---

## Session 2: Static vs Contextual Embeddings

---

## Session 2 - Question 1
**Type:** multiple-choice
**Time:** start
**Question:** In the Word2Vec analogy "king - man + woman = ?", what is the expected result?
- [ ] Prince
- [ ] Queen
- [ ] Princess
- [ ] Monarch
**Correct Answer:** Queen
**Explanation:** Word2Vec embeddings capture semantic relationships as vector arithmetic. The vector offset for "king→queen" is approximately the same as "man→woman". This works because words with similar contexts have similar vectors.
**Connection:** This analogy task is a fun way to demonstrate that static embeddings do capture real semantic relationships, even though they miss context-dependent meaning.

---

## Session 2 - Question 2
**Type:** multiple-choice
**Time:** during
**Question:** Why does the word "bank" need different embeddings in different sentences?
- [ ] Because it is spelled differently in different contexts
- [ ] Because it has different meanings (polysemy) depending on context
- [ ] Because Word2Vec cannot handle long words
- [ ] Because it is a rare word
**Correct Answer:** Because it has different meanings (polysemy) depending on context
**Explanation:** "Bank" can mean a financial institution or a river bank. Static embeddings assign one vector to "bank" regardless of meaning. Contextual embeddings produce different vectors based on surrounding words, solving this polysemy problem.
**Connection:** African languages are rich in polysemy and tonal distinctions. A word like "igba" in Yoruba has multiple meanings distinguished only by tone and context — exactly the kind of nuance contextual embeddings can capture.

---

## Session 2 - Question 3
**Type:** multiple-choice
**Time:** during
**Question:** What is the key difference between Word2Vec and BERT embeddings?
- [ ] Word2Vec is faster at inference
- [ ] BERT produces one vector per word regardless of context; Word2Vec varies by context
- [ ] Word2Vec produces one vector per word (static); BERT produces different vectors per context
- [ ] BERT was developed before Word2Vec
**Correct Answer:** Word2Vec produces one vector per word (static); BERT produces different vectors per context
**Explanation:** This is the fundamental distinction. Word2Vec learns a single embedding per word type. BERT (a Transformer) generates embeddings per token occurrence, conditioned on the entire input sequence via self-attention.
**Connection:** Understanding this distinction is crucial for choosing the right embedding approach for cohort projects. Simple similarity tasks may only need Word2Vec; nuanced understanding tasks (Q&A, translation) need contextual embeddings.

---

## Session 2 - Question 4
**Type:** multiple-choice
**Time:** during
**Question:** What neural network architecture powers BERT and other contextual embedding models?
- [ ] Recurrent Neural Network (RNN)
- [ ] Convolutional Neural Network (CNN)
- [ ] Transformer
- [ ] Generative Adversarial Network (GAN)
**Correct Answer:** Transformer
**Explanation:** BERT is built on the Transformer architecture, specifically the encoder stack. The key innovation is self-attention, which allows each token to attend to all other tokens in the sequence, producing context-aware representations.
**Connection:** The Transformer was introduced in the "Attention is All You Need" paper (Vaswani et al., 2017). It has become the foundation of modern NLP, including the Gemma models from Google DeepMind that we use in this cohort.

---

## Session 2 - Question 5
**Type:** poll
**Time:** during
**Question:** Which application of embeddings is most relevant to your project?
- [ ] Text classification (e.g., spam detection, topic labeling)
- [ ] Semantic search / retrieval
- [ ] Translation or multilingual NLP
- [ ] Question answering or chatbot
**Correct Answer:** N/A (poll — share results)
**Explanation:** This poll helps the instructor understand which downstream tasks students care about most, so they can tailor the remaining session time and office hours accordingly.
**Connection:** Maps to project domains: Education (55%) may focus on Q&A/chatbots; Healthcare (25%) on classification; Climate/Agriculture (20%) on search and translation.

---

## Session 2 - Question 6
**Type:** open-ended
**Time:** end
**Question:** How would you use word or sentence embeddings in your TRI AI project? What kind of text data do you have, and what do you want the embeddings to capture?
**Correct Answer:** (Open-ended — evaluate for thoughtfulness)
**Explanation:** Key criteria: identifying the data type (curriculum text, medical notes, farmer reports), the task (classification, search, generation), and why embeddings are suitable (vs. rule-based or bag-of-words approaches).
**Connection:** This is the most important question for the cohort — it forces students to connect the theoretical concepts of Session 2 to their own project work. Encourage students to be specific about their data and goals.

---

## Session 2 - Question 7
**Type:** poll
**Time:** end
**Question:** Session 2 rating — How would you rate today's session?
- [ ] 1 — Needs improvement
- [ ] 2 — Below average
- [ ] 3 — Average
- [ ] 4 — Good
- [ ] 5 — Excellent
**Correct Answer:** N/A (poll — feedback)
**Explanation:** End-of-session feedback helps the TRI AI team iterate on content delivery. Pay special attention to written comments if this poll is conducted via a platform that supports them.
**Connection:** Feedback from Cohort 10 directly shapes how future cohorts are taught. Every session builds on the last.
