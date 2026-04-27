# Assistant-axis projection — methodology matters

Two approaches to projecting CSPs onto the
[Butanium gemma-3-4b-it assistant axis](https://huggingface.co/datasets/Butanium/gemma-3-4b-it-assistant-axis)
yield **opposite** conclusions:

| | csp_arithmetic | csp-div (response-token) |
| --- | --- | --- |
| What's measured | L17 at the **CSP token positions** of the spliced input | L17 at **generated response tokens** |
| Aggregation | mean across 4 CSP tokens, then across prompts | mean across response tokens, then prompts |
| Subtracted baseline? | none — raw L17 vector | yes — shift = csp − vanilla |
| Sample | 65 persona CSPs (csp_arithmetic) | 10 KL-max CSPs (csp-div early_stop) |
| Mean cos | **−0.755** | **+0.310** |
| Range | [−0.759, −0.752] | [+0.07, +0.68] |
| n positive | **0 / 65** | **10 / 10** |

The csp_arithmetic-style measurement gives an extremely tight uniform-negative
result — every CSP, whether persona-trained or KL-max-trained, sits at
cos ≈ −0.76 with the assistant axis at L17.

## Replication on csp-div seeds

Re-running the csp_arithmetic methodology on our 10 early_stop seeds (KL ≈ 10):

| seed | cos | KL |
| -: | -: | -: |
| 0 | −0.7587 | 10.27 |
| 1 | −0.7571 | 10.31 |
| 2 | −0.7561 | 10.74 |
| 3 | −0.7583 | 10.06 |
| 4 | −0.7590 | 10.06 |
| 5 | −0.7581 | 10.44 |
| 6 | −0.7567 | 10.43 |
| 7 | −0.7571 | 10.11 |
| 8 | −0.7581 | 10.13 |
| 9 | −0.7579 | 10.33 |

Mean **−0.7577**, range **[−0.759, −0.756]** — essentially identical to
csp_arithmetic's persona CSPs. The training objective (persona-distillation
vs KL-maximization) is invisible in this measurement.

## Reconciling the two views

Both numbers are correct readings of different things:

- **csp_arithmetic-style**: at the CSP token positions, the model's L17
  state is uniformly negative on the axis. The CSP, as it sits in the
  user's input string, is a role-play-aligned representation.

- **csp-div (response-token shift)**: at the model's *response* tokens,
  the shift caused by the CSP is positively aligned with the axis (toward
  default-assistant direction, away from role-play).

These are consistent: **the CSPs are role-play-aligned vectors at input
position, but the model's response is still in the assistant register.**
The model partially refuses to inhabit the role and produces stage-directed
narration of a character (assistant voice, third-person observer of a
character) rather than first-person committed character speech. The
+0.31 mean response shift is the residue of that compromise.

## Why the csp_arithmetic cos is so uniform

The −0.76 value is suspiciously consistent across CSPs that differ
substantially in surface behavior (pirate vs analytical vs
divergent-KL-max). This suggests the cos is dominated by **structural
position** — being a vector at a user-input slot in the chat template —
rather than by content. At L17, the chat-template residual carries a
strong "this is user input, not assistant output" signal that always
projects in the role-play direction.

Test for this hypothesis: compute the cos of vanilla user-content L17
activations (any token in the user prompt) onto the same axis. If those
also sit at ~−0.76, the CSP projection is structural and tells us nothing
specific about the CSP's content. If vanilla is at 0 or positive, the
−0.76 is genuinely about CSP content.

## Implications

- The csp_arithmetic finding "assistant axis correlates with CSP-PCA"
  remains a real result — at the CSP token positions, all CSPs are at
  cos ≈ −0.76 and small differences in *scalar* projection (or ‖L17‖)
  may correlate with PC1.
- The csp-div response-token methodology measures something different
  and arguably more behaviorally relevant: how the model's *output*
  shifts under the CSP. That signal is positive, and reflects the
  "assistant narrating a character" register the CSPs actually produce.
- For questions about "is the CSP role-play-encoded?" — the answer
  depends on which level you measure at. The CSP is in input space; the
  model decides at output what to make of it.

## Negative control: vanilla L17 also sits at cos ≈ −0.75

To test the structural hypothesis, projected vanilla L17 (no CSP, no system
prompt) onto the same axis, at multiple positions:

| position (vanilla) | cos |
| - | -: |
| user-content span mean | −0.7524 |
| last input token | −0.7514 |
| first input token (BOS) | −0.7523 |
| chat-template pre-user (system header tokens) | −0.7516 |
| chat-template post-user (model marker) | −0.7506 |

Every position projects at cos ≈ −0.75 — within 0.005 of the CSP results.
**The −0.76 cos is fully accounted for by chat-template structural signal**;
the CSPs aren't pulling toward role-play, they're just sitting at the same
"L17-of-Gemma-3-4b-on-chat-input" baseline as vanilla.

## Extending to csp_arithmetic personas

| population | n | cos_mean | cos_std | cos range |
| - | -: | -: | -: | -: |
| vanilla (5 positions) | — | −0.752 | — | [−0.752, −0.751] |
| persona CSPs (roles) | 33 | −0.7561 | 0.0017 | — |
| persona CSPs (traits) | 32 | −0.7543 | 0.0013 | — |
| persona CSPs (all) | 65 | −0.7552 | 0.0017 | [−0.7585, −0.7523] |
| KL-max CSPs (csp-div) | 10 | −0.7577 | 0.0009 | [−0.7590, −0.7561] |

All CSPs are within ±0.002 of each other and within ±0.005 of vanilla.
Training objective (persona-distillation vs KL-max) is invisible at this
scale.

## Tracking IS real at the fine scale

Despite the tiny absolute spread, PC1 of CSP embeddings rank-correlates
strongly with axis-cos:

| population | n | Spearman(PC1, axis_cos) | p |
| - | -: | -: | -: |
| csp_arithmetic personas | 65 | **−0.806** | <0.001 |
| csp-div KL-max CSPs | 8 | **+0.786** | 0.021 |

(PC1 sign is arbitrary in PCA, so opposite signs across populations is fine
— the magnitudes are the relevant comparison.)

So the axis IS picking up content variation, but at a magnitude swamped by
the structural "user-input-position" baseline. The csp_arithmetic claim
"CSPs track along the assistant axis" is correct in the rank-order sense,
even though every CSP and every vanilla position cluster tightly around
cos = −0.755.

## Reconciliation summary

There are now four numbers describing the same axis projection, each at
a different level of analysis:

| measurement | mean cos | what it captures |
| - | -: | - |
| vanilla L17 anywhere on chat-template input | −0.751 | structural template signal |
| raw L17 at CSP token positions | −0.756 | the same + tiny CSP-content variation |
| shift = CSP-tokens − vanilla user-span | +0.74 | mostly artifact of length-of-aggregation difference |
| shift = response tokens − vanilla response | +0.31 | how the *generation* differs (the only behaviorally-relevant signal) |

The most useful number for "what does the CSP do behaviorally" is the
last one — and there, all 10 KL-max CSPs project positively, consistent
with the model producing assistant-narrating-a-character output rather
than fully inhabited role-play.

## Files

- `assistant_axis.png` / `.json` — response-token-shift methodology (csp-div, primary)
- `assistant_axis_csptoken.json` — csp_arithmetic-style replication on csp-div seeds
- `assistant_axis_vanilla_control.json` — vanilla L17 at 5 positions (negative control)
- `assistant_axis_shift.json` — shift = CSP-tokens − vanilla user-span, persona + KL-max CSPs
- Original csp_arithmetic cache at
  `/tmp/csp_arithmetic/results/pca/axis_projection_cache.json` (65 personas)
