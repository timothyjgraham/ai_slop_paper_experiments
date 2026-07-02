# Data dictionary

## Embeddings (`experiments/empirical/embeddings/`)

The analysis runs directly from these precomputed embeddings, so all reported
numbers are reproducible without regenerating the image corpus.

### `embeddings.npz` (CLIP)
A compressed NumPy archive with two arrays:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `embeddings` | (1177, 768) | float32 | CLIP ViT-L/14 image embeddings, one row per analysed image |
| `metadata`   | scalar (JSON string) | str | A JSON-encoded list of 1177 records, aligned row-for-row with `embeddings` |

Each metadata record has the following fields:

| Field | Example | Description |
|-------|---------|-------------|
| `path` | `images/openai/cultural_identity/ci_american_000.png` | Relative path to the source image (paths are repository-relative; the image files themselves are not redistributed) |
| `prompt_id` | `ci_american` | Identifier matching an entry in `prompts.json` |
| `category` | `cultural_identity` | One of `cultural_identity`, `open_ended`, `artistic_style` |
| `platform` | `openai` | `openai` (DALL-E 3) or `google` (Imagen 4) |
| `group` | `majority` | `majority` or `minority` cultural group |
| `nationality` | `American` | Nationality/culture targeted by the prompt |

### `embeddings_dinov2.npz` (DINOv2, robustness replication)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `embeddings` | (1177, 1024) | float32 | DINOv2 image embeddings (self-supervised; used to confirm findings are not CLIP-specific) |
| `valid_indices` | (1177,) | int64 | Row alignment back to the CLIP metadata order |

## Composition of the analysed set (N = 1177)

| Split | Count |
|-------|-------|
| OpenAI / DALL-E 3 | 598 |
| Google / Imagen 4 | 579 |
| cultural_identity | 588 |
| open_ended | 394 |
| artistic_style | 195 |

1,200 images were designed (30 prompts × 20 images × 2 models); 1,198 generated
successfully; 1,177 retained after removing failed or invalid generations.

## Ablation embeddings (`experiments/empirical/ablation/`)

### `ablation_embeddings.npz`
One array per condition, keyed `"{model}__cfg{scale}__{prompt_id}"` (e.g.
`sdxl-base__cfg7.5__ci_american`), each of shape (10, 768) float32: CLIP
ViT-L/14 embeddings of the 10 images generated for that condition.
88 conditions = 2 models (`sdxl-base`, `sdxl-dpo`) × 4 CFG scales
(1.0, 5.0, 7.5, 15.0) × 11 prompts = 880 images.

### `ablation_index.json`
A list of records aligned with the archive keys, each with fields `model`,
`cfg`, `prompt_id`, `key`, and `n`.

## Prompts (`experiments/empirical/prompts.json`)
30 prompts across the three categories above, each annotated with its category,
group (majority/minority), and target nationality. See the file's `metadata`
block for the mapping to the paper's theorems and predictions.
