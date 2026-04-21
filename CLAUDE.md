# SC4001 Final Project — Oxford Flowers 102 (Project F)

NTU SC4001 (Neural Networks and Deep Learning) solo final project. Project F chosen 2026-04-21. Deadline 2026-04-19 (per handout). Scoping in progress — no code yet.

## Problem shape

102-way fine-grained flower classification with **10 training images per class** (1020 train / 1020 val / 6149 test). Test set is larger than train, which forces transfer learning or parameter-efficient tuning from a strong pretrained backbone. Training from scratch is a dead end.

## Context files (read these before working)

All context lives in the Obsidian vault at `/Users/truonghuy/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/`.

- **Synthesis — option comparison and decision rationale:**
  `llm-wiki/wiki/syntheses/sc4001-project-options.md`
  (see the "F. Flowers Recognition (Oxford Flowers 102)" section and the "Follow-ups" section at the bottom for open scoping questions)

- **Source — course handout (assessment weights, rules, deliverables):**
  `llm-wiki/wiki/sources/sc4001-project-handout.md`

- **Source — instructor FAQ clarifications (page limit, library policy, novelty bar):**
  `llm-wiki/wiki/sources/sc4001-project-faq.md`

## Key constraints (from handout + FAQ)

- Solo project (not a 3-person team); scope must fit one person.
- 10-page PDF report, Arial 10, excluding references/cover/content/appendix.
- Assessment: execution 30% · experiments 30% · report 15% · novelty 15% · peer review 10%.
- Must train from scratch or fine-tune with **original innovation**. Running a pretrained model end-to-end without adaptation is disallowed.
- Open-source libraries allowed with citation.
- Late penalty: 5%/day, up to 3 days.

## Open scoping questions (from synthesis follow-ups)

1. Backbone choice: ViT-B/16 (ImageNet-21k) vs CLIP ViT.
2. PEFT method set: linear probe, full FT, VPT-Shallow/Deep, LoRA, adapters — which subset, matched how.
3. Shot settings: 1/5/10/20 per class.
4. Ablation axes: augmentation (none / RandAug / MixUp / CutMix), loss (CE / triplet / ArcFace).
5. Whether a bake-off alone clears the 15% novelty bar, or a distinctive twist is needed (prompt-length schedule, MixUp in prompt space, triplet+VPT combo, class-conditioned MixUp).

## Risks to watch (from synthesis)

- "I loaded CLIP and got 97%" trap — story must be trends across methods/shots/losses, not a headline number.
- CLIP pretraining may be contaminated with flower imagery; note the caveat.
- VPT prompt placement: shallow = input only, deep = every layer, no position embeddings on prompts.
- Triplet loss needs semi-hard mining.
- Don't train on val.
