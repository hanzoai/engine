# EmbeddingGemma 2: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)

EmbeddingGemma 2 is Google DeepMind's multilingual embedding model (Apache-2.0), built on Gemma 4. It maps text
(and, in the reference model, images, video and audio) into one 768-dimensional space, trained with Matryoshka
representation learning so the leading 512, 256 or 128 dimensions are embeddings too.

Hanzo Engine serves its text backbone (`EmbeddingGemma2Model`'s `language_model`: 24 layers, hidden 512, 270M
parameters with the 262,144-token embedding table) followed by sentence-transformers' mean pooling and
normalization. The vision and audio towers are not loaded: `/v1/embeddings` takes text. An input holds at most
8,192 tokens, the model card's context window.

For a catalog of embedding models, see [EMBEDDINGS.md](EMBEDDINGS.md).

## Quick start

```bash
hanzo serve embedding -m google/embeddinggemma-2 -p 1234
```

The architecture is detected from `config.json` (`EmbeddingGemma2Model`); pass `-a embeddinggemma2` to name it.
Run it in F32 or BF16, never F16: its activations exceed F16's range. The CPU's default is F32.

## Prompts

The model was trained with task prefixes, which the caller writes into the text; the server embeds the text as
given. Queries and documents of a retrieval task take different prefixes; symmetric tasks put the same one on
every input.

| Use | Query | Document |
| --- | --- | --- |
| Search | `task: search result \| query: {query}` | `title: {title or none} \| text: {content}` |
| Question answering | `task: question answering \| query: {question}` | `title: {title or none} \| text: {passage}` |
| Fact checking | `task: fact checking \| query: {claim}` | `title: {title or none} \| text: {evidence}` |
| Code search | `task: code retrieval \| query: {query}` | `title: {title or filename} \| text: {code}` |
| Classification | `task: classification \| query: {content}` | |
| Clustering | `task: clustering \| query: {content}` | |
| Similarity | `task: sentence similarity \| query: {content}` | |

## Shorter embeddings

`dimensions` keeps the leading values and normalizes them again, as sentence-transformers' `truncate_dim` with
`normalize_embeddings` does. A query and the documents it is compared with must share a width.

```bash
curl http://localhost:1234/v1/embeddings -H "Content-Type: application/json" -d '{
  "model": "default",
  "input": ["task: search result | query: What causes the northern lights?"],
  "dimensions": 256
}'
```

## Beside another model

One process serves several embedding models in multi-model mode; a request picks one by `model`:

```json
{
  "Qwen/Qwen3-Embedding-0.6B": {
    "Embedding": { "model_id": "Qwen/Qwen3-Embedding-0.6B", "arch": "qwen3embedding" },
    "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
  },
  "google/embeddinggemma-2": {
    "Embedding": { "model_id": "google/embeddinggemma-2", "arch": "embeddinggemma2" },
    "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b"
  }
}
```

`revision` pins the weights: an index holds the vectors of one revision, and `main` moves.

```bash
hanzo-server --port 1234 multi-model --config models.json --default-model-id Qwen/Qwen3-Embedding-0.6B
```

## Parity

Against sentence-transformers 6.1.0 and transformers 5.19.0 in F32 on the CPU, one text at a time, over 64 texts
in 14 languages, code and the prompts above, four of them 250 to 1,256 tokens (past a sliding layer's reach of
513): every vector's cosine is at least 0.9999993 (`1 - cos` at most 6.5e-7), at 768, 512, 256 and 128
dimensions, and every token count equals the reference tokenizer's.

## License

google/embeddinggemma-2 is released by Google DeepMind under the Apache License 2.0
([model card](https://huggingface.co/google/embeddinggemma-2)).
