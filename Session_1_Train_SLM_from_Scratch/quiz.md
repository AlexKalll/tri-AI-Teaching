# Session Quiz — Train a Small Language Model (BPE + Embeddings)

These multiple-choice questions are for the live Zoom quiz during the session.
Each question has one correct answer (marked with **Answer:**) and a short
explanation you can use to talk it through.

---

## 1. What is the main job of a tokenizer in an LLM pipeline?
- A) To generate new text
- B) To convert raw text into tokens the model can process
- C) To compute cosine similarity
- D) To train the model's weights

**Answer: B** — The tokenizer turns human text into the numbered tokens the model actually reads.

## 2. How does Byte-Pair Encoding (BPE) build its vocabulary?
- A) It assigns exactly one token per character
- B) It starts with characters and repeatedly merges the most frequent adjacent pair
- C) It uses a fixed dictionary of 1000 words
- D) It randomly splits words apart

**Answer: B** — BPE begins with characters, then merges the most common pairs step by step to form subwords.

## 3. Why do modern tokenizers use subwords (like "learn" + "ing") instead of whole words?
- A) Subwords make the model slower
- B) They handle rare and unseen words by composing familiar pieces
- C) They remove the need for embeddings
- D) They force the vocabulary to grow into the millions

**Answer: B** — Breaking words into subwords lets the model understand new words from pieces it already knows.

## 4. What is a word embedding?
- A) A random list of word IDs
- B) A dense vector that captures a word's meaning as numbers
- C) The raw text of a sentence
- D) A type of attention mask

**Answer: B** — An embedding is a list of numbers (a vector) where nearby vectors mean similar meanings.

## 5. Cosine similarity measures the angle between two embedding vectors. A value close to 1 means the two words are:
- A) Unrelated
- B) Opposites
- C) Very similar in meaning
- D) Exactly the same word

**Answer: C** — A cosine near 1 means the vectors point in nearly the same direction → similar meaning.

## 6. A cosine similarity close to 0 between two words tells us they are:
- A) Nearly identical
- B) Unrelated / orthogonal in meaning
- C) Translations of each other
- D) Always next to each other in text

**Answer: B** — A value near 0 means the vectors are at right angles, so the words are not semantically related.

## 7. In our tiny model, increasing "block_size" (the context window) from 8 to 32 mainly helps the model:
- A) Use less memory
- B) See more of the story before predicting the next token
- C) Remove all punctuation
- D) Train instantly

**Answer: B** — A bigger context window lets the model look further back before guessing the next word.

## 8. Increasing "n_embd" (embedding dimension) from 32 to 64 gives each token:
- A) A shorter word ID
- B) A richer vector representation
- C) A fixed position in the sentence
- D) Fewer total parameters

**Answer: B** — More dimensions = a more detailed vector, so each word can capture more nuance.

## 9. What does training a small language model actually learn?
- A) It memorizes sentences word-for-word
- B) It learns better embeddings (and weights) to predict the next token
- C) It deletes rare words from memory
- D) It randomly rearranges the tokens

**Answer: B** — Training adjusts the weights so the model gets better at predicting what comes next.

## 10. In next-token prediction, what is the model trying to output at each step?
- A) A summary of the whole text
- B) The most likely next token given the context
- C) A completely random word
- D) The very first token again

**Answer: B** — The model scores every possible next token and picks the most likely one.

## 11. What does the attention mechanism let the model do?
- A) Ignore the context completely
- B) Weight different context tokens by how relevant they are
- C) Shrink the vocabulary size
- D) Replace the embeddings entirely

**Answer: B** — Attention lets the model focus more on the tokens that matter for the next prediction.

## 12. When you scale the same model recipe to more dimensions and more layers, what generally happens?
- A) The parameter count stays exactly the same
- B) The model ends up with fewer parameters
- C) The parameter count and capacity grow (e.g., ~12.9M parameters)
- D) Training becomes instant

**Answer: C** — More dimensions and layers mean many more parameters, which lets the model capture more patterns.

---

### Answer key (quick reference)
1. B · 2. B · 3. B · 4. B · 5. C · 6. B · 7. B · 8. B · 9. B · 10. B · 11. B · 12. C
