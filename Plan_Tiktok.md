# Inside TikTok's Architecture

> A technical overview of the real-time data pipelines, distributed memory systems, vector retrieval, and machine-learning ranking used by large-scale recommendation platforms such as TikTok.

## Table of Contents

1. [High-Level AI Factory Architecture](#1-high-level-ai-factory-architecture)
2. [Real-Time Log Pipeline](#2-real-time-log-pipeline)
3. [Monolith and Cuckoo Hashing](#3-monolith-and-cuckoo-hashing)
4. [Candidate Retrieval with Two-Tower Networks](#4-candidate-retrieval-with-two-tower-networks)
5. [Fast Approximate Vector Search](#5-fast-approximate-vector-search)
6. [Fine Ranking with MMoE](#6-fine-ranking-with-mmoe)
7. [Lifecycle of a User Profile Vector](#7-lifecycle-of-a-user-profile-vector)
8. [End-to-End Architecture Summary](#8-end-to-end-architecture-summary)

---

## 1. High-Level AI Factory Architecture

Unlike traditional social networks that depend heavily on a fixed social graph, large-scale recommendation systems convert each interaction into a signal that can update the next feed request.

```text
+-----------------------+       +-------------------------+       +----------------------+
| Video delivery        |       | Real-time behavior      |       | Recommendation       |
| - Transcoding         |       | - Dwell time            |       | - Candidate retrieval|
| - Feature extraction  |       | - Replays               |       | - Fine ranking       |
| - CDN distribution    |       | - Swipes and hesitation |       | - Feed composition   |
+-----------------------+       +-------------------------+       +----------------------+
           |                            |                              |
           +----------------------------+------------------------------+
                                        |
                                        v
                           Personalized For You feed
```

### Core components

| Component                 | Purpose                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| High-speed video pipeline | Transcodes uploaded media, extracts features, creates content embeddings, and caches assets close to users. |
| Streaming data engine     | Collects interaction events and updates user or model state with low latency.                               |
| Recommendation engine     | Retrieves a small candidate set and applies multi-objective ranking before returning the feed.              |

### Two-stage recommendation flow

```text
Millions of videos
        |
        v
Deep retrieval / two-tower matching
        |
        v
Hundreds of candidates
        |
        v
Multi-objective fine ranking
        |
        v
Final ordered feed
```

---

## 2. Real-Time Log Pipeline

A decoupled streaming pipeline allows the feed to respond to behavior without blocking the user-facing application during traffic spikes.

```text
TikTok app
    |
    | HTTPS interaction event
    v
+--------------------+
| Flume / Scribe     |  Collect and format edge logs
+--------------------+
    |
    v
+--------------------+
| Apache Kafka       |  Buffer and distribute event streams
+--------------------+
    |
    v
+--------------------+
| Apache Flink       |  Join and process live events
+--------------------+
    |
    v
+--------------------+
| Parameter server   |  Update user preference state
+--------------------+
    |
    v
Personalized next feed
```

### Event-flow stages

| Stage             | Responsibility                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| Collection        | Edge agents collect, structure, and forward interaction logs.                                                   |
| Transport         | Kafka distributes high-volume events between producers and consumers while absorbing spikes.                    |
| Stream processing | Flink joins interaction events with content context and continuously computes state updates.                    |
| Serving           | The recommendation system reads updated state when producing the next request's candidates and ranking results. |

### Commonly collected signals

- Explicit interactions, such as likes, comments, shares, and follows.
- Implicit interactions, such as watch duration, completion, replay, swipe speed, and hesitation.
- Context, such as device, session state, location when permitted, language, and active viewing sequence.

---

## 3. Monolith and Cuckoo Hashing

Recommendation systems maintain very large sparse embedding tables. Standard hash tables can place different keys into the same slot, while fixed hashing strategies may make collisions expensive to resolve.

**Monolith**, ByteDance's recommendation architecture, uses a parameter-serving model designed for rapidly changing sparse embeddings.

### Cuckoo hashing

Cuckoo hashing stores each key in one of two possible locations selected by separate hash functions.

```text
Key A chooses T0
      |
      v
T0 slot is occupied by B
      |
      v
Move B to its alternative location in T1
      |
      v
Move the displaced key to its alternative location
      |
      v
Continue until an empty slot is found
```

### Important properties

- Each key has two potential locations.
- Lookup generally compares only the two candidate locations.
- Insertion evicts and relocates an occupant when both locations are full.
- Long eviction chains require rebuilding or resizing the table when load becomes too high.
- Expiration and filtering reduce stale entries and active memory usage.

### Memory optimization techniques

| Technique                | Purpose                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------- |
| Count-Min Sketch         | Estimates item frequency and helps identify cold or low-value entries.              |
| Time-to-live             | Automatically expires stale preference fields.                                      |
| Sharding                 | Splits the parameter space across independent server processes or machines.         |
| Sparse embedding storage | Stores parameters for observed IDs instead of maintaining dense parameter matrices. |

---

## 4. Candidate Retrieval with Two-Tower Networks

Scanning all videos and running a deep ranking model for every candidate would violate the latency budget. Candidate retrieval first narrows the corpus with two independent embedding towers.

```text
User ID + context                     Video ID + metadata
       |                                      |
       v                                      v
+---------------+                      +---------------+
| User tower    |                      | Item tower    |
+---------------+                      +---------------+
       |                                      |
       v                                      v
User embedding u                     Video embedding v
       |                                      |
       +-------------- dot product ------------+
                       |
                       v
                Similarity score
```

### User tower

The user tower processes:

- Current session activity.
- Recent interaction sequence.
- User demographics and language, when permitted and available.
- Device or environmental context.

It produces a dense vector representing the user's current interest state.

### Item tower

The item tower processes content features such as:

- Visual features extracted by computer vision models.
- Speech or captions processed with language models.
- Audio features and music metadata.
- Creator, topic, and engagement features.

It produces a dense vector representing each candidate video.

### Decoupled inference

The item tower can run offline when a video is published or updated. The resulting vectors can be placed in an ANN index. The live request only:

1. Computes the user vector with the user tower.
2. Searches the precomputed item index.
3. Returns a candidate set, typically a few hundred videos.
4. Sends those candidates to the fine-ranking model.

This separation keeps expensive model execution off the retrieval request path.

---

## 5. Fast Approximate Vector Search

Candidate retrieval uses approximate nearest-neighbor search rather than exact comparison against every video.

To understand how FAISS (Facebook AI Similarity Search) and ScaNN (Scalable Nearest Neighbors by Google) work, it helps to look at the exact computational problem they solve.

![alt text](image.png)

### FAISS inverted-file indexing

FAISS can cluster vectors and build an inverted-file index:

1. Group vectors around learned centroids using clustering.
2. Assign each content vector to a cluster.
3. Search the clusters closest to the user vector.
4. Compare only vectors belonging to those clusters.

This trades a small amount of retrieval recall for substantially lower query cost.

### ScaNN anisotropic quantization

ScaNN improves quantized-vector similarity by reducing errors in directions that matter more for nearest-neighbor ranking. It combines:

- Learned partitioning.
- Anisotropic vector quantization.
- Efficient scoring for candidate retrieval.

### Product Quantization

Product Quantization compresses vectors by splitting them into smaller sub-vectors and replacing each segment with a codebook index.

```text
512-dimensional float vector
            |
            v
Split into 64 sub-vectors of 8 dimensions
            |
            v
Replace each segment with its nearest codebook entry
            |
            v
[42, 119, 7, ..., 211]  (64 bytes)
```

### Search process

1. Split the live query vector into the same number of segments as the stored vectors.
2. Compare each segment against its corresponding codebook.
3. Store the resulting distances in a lookup table.
4. Score a compressed candidate by summing its codebook lookups.
5. Avoid decompressing full vectors during the search.

This converts high-cost floating-point comparisons into compact storage and table lookups.

### Comparison

| Method               | Best-known role                     | Main trade-off                                              |
| -------------------- | ----------------------------------- | ----------------------------------------------------------- |
| Exact search         | Highest possible recall             | Too expensive at global scale.                              |
| FAISS IVF            | Fast coarse cluster filtering       | Candidate recall depends on how many clusters are searched. |
| ScaNN                | Accurate approximate retrieval      | Requires more complex index configuration and training.     |
| Product Quantization | Memory-efficient compressed vectors | Approximation can reduce retrieval precision.               |

---

## 6. Fine Ranking with MMoE

Fine ranking evaluates a few hundred candidates and models interactions between features. A Multi-Gate Mixture-of-Experts model can learn shared patterns while allowing separate tasks to emphasize different experts.

```text
                    Candidate features
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
          Expert 1      Expert 2  ...  Expert N
              |             |             |
              +----- shared representations ----+
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
      Like gate        Watch gate         Share gate
          |                 |                 |
          v                 v                 v
 P(Like) | E(Watch time) | P(Share)
```

### MMoE components

| Component           | Purpose                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| Experts             | Specialized neural sub-networks learn patterns from candidate and user features. |
| Task-specific gates | Softmax gates choose how much each task should use each expert.                  |
| Task towers         | Convert expert representations into predictions for each objective.              |
| Blending function   | Converts task predictions into one final score and feed order.                   |

### Example blended score

$$
	ext{Final Score} = w_1 P(Like) + w_2 P(Share) + w_3 E(WatchDuration) - w_4 P(	ext{Repetitive})
$$

Weights are adjusted to support platform goals such as:

- Longer watch duration.
- More meaningful social interactions.
- Creator diversity.
- Reduced repetitive recommendations.
- Content safety and policy constraints.

---

## 7. Lifecycle of a User Profile Vector

A user profile is not a single static embedding. It commonly combines stable identity, long-term interests, and short-term intent.

| Layer                 | Typical time scale | Purpose                                                 |
| --------------------- | ------------------ | ------------------------------------------------------- |
| Static profile        | Months to years    | Language, device, demographics, and account attributes. |
| Long-term preferences | Weeks to months    | Stable topics and recurring interests.                  |
| Real-time intent      | Seconds to minutes | Recent viewing sequence and immediate context.          |

### Real-time update flow

```text
Recent user actions
        |
        v
+-----------------------+
| Short-term attention  |  Fast updates
+-----------------------+
        |
        +--------------------+
                             |
Recent actions             Stable preferences
        |                             |
        +-------------+---------------+
                      |
                      v
             Updated user vector
                      |
                      v
             Next candidate search
```

### Preventing rapid profile whiplash

A one-off interaction should not erase the user's stable interests. A common design uses two update gates:

- A **short-term gate** reacts quickly to a recent interaction or viewing sequence.
- A **long-term gate** changes slowly unless patterns are repeated over time.

For example, a short cat-video session may temporarily add cat videos to the next candidate set without changing the user's long-term interests.

### State synchronization

```text
User action
    |
    v
Interaction log
    |
    v
Stream processor
    |
    v
Feature and embedding state
    |
    v
Online parameter server
    |
    v
Retrieval requests read the latest state
```

---

## 8. End-to-End Architecture Summary

```text
+---------------------+       +----------------------+
| Upload and content  | ----> | Feature extraction   |
| transcoding         |       | and embedding        |
+---------------------+       +----------------------+
                                      |
                                      v
                              +---------------+
                              | Video index   |
                              +---------------+
                                      |
User action -> behavior logs -> stream processing -> profile state
                                                      |
                                                      v
                                           User embedding
                                                      |
                                                      v
                                           ANN candidate search
                                                      |
                                                      v
                                              MMoE ranking
                                                      |
                                                      v
                                             Ordered feed
```

## Design principles

1. Keep the hot serving path small and predictable.
2. Precompute expensive item embeddings.
3. Decouple ingestion, stream processing, serving, and training.
4. Retrieve candidates approximately, then rank them accurately.
5. Separate short-term intent from stable preferences.
6. Measure latency, recall, diversity, ranking quality, and system cost together.
7. Apply safety, privacy, policy, and abuse controls to data collection and ranking.

---

_This document describes a general large-scale recommendation architecture. TikTok's proprietary implementation details may differ._
