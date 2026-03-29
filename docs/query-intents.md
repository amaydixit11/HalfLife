# Query Temporal Intents

HalfLife distinguishes between three main query "intents" to determine how aggressively to weight temporal signals.

---

## 🕒 1. **Fresh Intent**
- **Keywords**: `"latest"`, `"recent"`, `"newest"`, `"current"`, `"today"`, `"breaking"`, `"just released"`
- **Goal**: Find the absolute most up-to-date information.
- **Scoring**:
  - High temporal weight ($\beta \approx 0.6$)
  - Lower vector weight ($\alpha \approx 0.3$)
  - Small trust weight ($\gamma \approx 0.1$)
- **Effect**: Even if a 3-year-old paper has a slightly higher semantic similarity, a paper from last week will rank significantly higher.

---

## 📜 2. **Historical Intent**
- **Keywords**: `"history of"`, `"origins"`, `"background"`, `"how did X evolve"`, `"evolution"`, `"background information"`
- **Goal**: Surface the foundational, early documents that define a topic's roots.
- **Scoring**:
  - Moderate temporal weight ($\beta \approx 0.3$)
  - **Inversion Logic**: Temporal score is calculated as $1.0 - \text{decay}(\Delta t)$. This literally rewards older documents.
- **Effect**: Queries like "history of transformer models" will prioritize the 2017 "Attention Is All You Need" paper over the latest "Llama 3" technical report.

---

## 🧱 3. **Static (Time-Agnostic) Intent**
- **Keywords**: `"define"`, `"what is"`, `"explain"`, `"how to"`, `"formula for"`
- **Goal**: Find the most semantically accurate answer regardless of when it was written.
- **Scoring**:
  - High vector weight ($\alpha \approx 0.7$)
  - Very low temporal weight ($\beta \approx 0.1$)
- **Effect**: Retrieval behavior reverts almost entirely to standard cosine similarity, with only a small bias toward trusted sources.

---

## 🛠 **Under the Hood: The Intent Classifier**
Located in `engine/classifier/query_intent.py`, this component uses keyword matching (and later a small transformer-based classifier) to label every query before it enters the `Reranker`. 

If a keyword triggers both intents(e.g., "latest history of X"), **Fresh** intent typically takes precedence as it implies a recent update to a historical topic.
