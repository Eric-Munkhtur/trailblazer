# Diagnosis of the two MSI HINT modifications

## Conclusion

Neither modification demonstrated a meaningful improvement in this single-seed, five-epoch continuation experiment. The new pooling layers were active and learned nonuniform weights. Their effects on the final predictions were small; the conditioned variant mostly shifted probabilities downward without improving ranking. No single root cause has been established.

The original hypothesis—that eligibility mean pooling was an important performance bottleneck—was not established before choosing this intervention. These results do not support that hypothesis strongly enough to claim a better predictor.

## Experimental scope

Three arms continue from the same authors' checkpoint, using the same cached BioBERT vectors and Phase II splits (4,005 train / 446 validation / 1,654 test). Each uses seed 0, batch 32, Adam learning rate 0.0003, five epochs, and selection by lowest validation BCE including epoch zero. All three selected epoch four. This compares matched continuations, not training from scratch or a replication of the paper's headline score.

The conditioned attempt was exploratory after inspecting the first attempt's test results. This limits confirmatory claims. Today's diagnosis used existing test predictions for descriptive analysis and bootstrapping. New model inference and counterfactual ablations used validation only. No training or tuning was performed.

## What changed

- **Sentence attention (global scoring):** replace the mean of each inclusion/exclusion sentence group with a softmax-weighted average using two learned 768-dimensional scoring vectors (1,536 added parameters). The scoring rule is shared across trials and has no drug/disease input.
- **Conditioned attention:** derive separate inclusion/exclusion queries from the concatenated drug and disease representations, using a shared 32-dimensional sentence-key space (31,040 added parameters). The original protocol projection and downstream HINT interaction graph remain.
- Both preserve mean pooling at initialization. Both retain frozen cached sentence embeddings; both ultimately compress eligibility into a 50-dimensional protocol vector. Neither adds biological evidence or PrimeKG paths to HINT.

## Measured outcomes

| Arm | Validation BCE | Test ROC-AUC | Test AP | Test F1 at 0.5 |
|---|---:|---:|---:|---:|
| Original HINT continuation | 0.621753 | 0.632018 | 0.682925 | 0.605844 |
| Global attention | 0.621373 | 0.632309 | 0.683349 | 0.605592 |
| Conditioned attention | 0.621629 | 0.630982 | 0.682502 | 0.596026 |

### 1. The new layers were working

Validation predictions from reloaded frozen checkpoints reproduced saved GPU probabilities within 1.20e-7. The code routes conditioned features through the inherited HINT graph forward method. Attention weights were nonuniform, and changing the drug/disease context changed the conditioned attention weights. This argues against a disconnected layer or weights remaining at uniform initialization. It does not constitute a proof that every possible implementation defect is absent.

### 2. Learned attention had limited influence on outcome predictions

On validation, temporarily restoring uniform pooling while retaining each modified model's learned downstream weights gave:

| Frozen model | Learned pooling AUC | Uniform pooling AUC | Mean absolute probability change |
|---|---:|---:|---:|
| Global attention | 0.712407 | 0.711235 | 0.008037 |
| Conditioned attention | 0.712064 | 0.711215 | 0.006779 |

These are inference counterfactuals, not separately trained baselines. They show a small local contribution from learned pooling, not a clean causal attribution of the entire training comparison.

For conditioned attention, exclusion weights remained close to uniform: average normalized entropy 0.992 (1 means uniform). The average cosine between its weighted raw sentence summary and the mean summary was 0.9959; the corresponding global-attention value was 0.9915. However, the subsequent projection amplified some differences: average relative L2 changes in protocol representations were 0.249 and 0.367 respectively. Thus it would be inaccurate to claim that the protocol representations never changed.

Rolling the drug/disease context to a different row within each validation batch changed attention weights, but changed outcome probabilities by only 0.00790 on average. Validation AUC became 0.710912, versus 0.712064 with the correct context. The context connection is functional; its measured predictive benefit here is small and metric-dependent (AP did not improve under the correct-context comparison).

### 3. Most predictions barely changed; the second model shifted scores downward

| Comparison against original, test set | Global | Conditioned |
|---|---:|---:|
| Probability correlation | 0.99887 | 0.99905 |
| Mean absolute probability change | 0.00757 | 0.01041 |
| Mean signed probability change | -0.00338 | -0.00929 |
| Wrong classifications corrected, threshold 0.5 | 8 | 5 |
| Correct classifications broken, threshold 0.5 | 10 | 15 |

The conditioned model made 743 positive predictions versus the control's 759. Its lower F1 is partly explained by this downward shift and the resulting threshold crossings. A constant score shift alone would not change ROC-AUC; its small AUC decline reflects changes in ordering as well.

### 4. There is no convincing improvement within sampling uncertainty

Paired stratified bootstrap of the 1,654 existing test predictions, 1,000 resamples, seed 20261011:

- Global AUC delta: +0.000290; percentile 95% interval [-0.001303, +0.001736].
- Conditioned AUC delta: -0.001036; interval [-0.002506, +0.000452].
- Both AP delta intervals also include zero; exact values are in diagnosis.json.

These intervals describe trial resampling for the fixed trained models. They do not measure training-seed uncertainty, adjust for exploratory model selection, or account for possible correlations among related trials. They support neither a robust improvement nor a definitive claim that attention always harms performance.

### 5. More epochs are not an established fix

All arms had their lowest validation BCE at epoch four, then worsened at epoch five while training loss fell. Conditioned training loss at epoch five was 0.620403 versus the original's 0.621398; validation loss was 0.623896 versus 0.623654. This is consistent with early overfitting and provides no evidence that simply extending these runs would deliver improvement.

### 6. An inherited parsing limitation and split differences

Upstream split_protocol only separates inclusion/exclusion when its heading-order condition succeeds. Otherwise protocol2feature uses the same complete group for both branches.

| Split | Trials | Duplicated unsplit criteria | Fraction | Mean parsed lines |
|---|---:|---:|---:|---:|
| train | 4005 | 570 | 14.23% | 39.23 |
| valid | 446 | 73 | 16.37% | 38.98 |
| test | 1654 | 28 | 1.69% | 49.37 |

There were zero missing cached lines and zero empty criteria across these splits. Thus missing embedding coverage does not explain this result. The parser limitation is shared by all arms; it does not invalidate their matched comparison. The substantial differences in parsing success and protocol length are observed distribution differences, but we have not established that they caused the attention results. The parser was not changed during diagnosis.

## Interpretation and responsibility

The evidence supports a narrow conclusion: these specific pooling interventions, with this checkpoint, data, seed, and training budget, did not improve generalization meaningfully. Plausible contributors include limited new information (reweighting fixed vectors), redundancy with HINT's existing downstream drug/disease/protocol interactions, and a weak outcome-supervision signal for learning sentence relevance from 4,005 trials. These are hypotheses, not individually proven causes.

A stronger process would have measured the importance of the protocol branch and sensitivity to pooling before spending the last training attempt on a more elaborate pooling rule. The GPU completed the work; additional hardware would not itself fix a weak modeling hypothesis. The correct research claim is a documented negative ablation, not a demonstrated improvement or a general verdict against attention.

## Reproducibility files

- analyze.py: exact diagnostic script run on MSI; requires the existing trusted checkpoint, source, assets, and saved runs at its recorded paths.
- diagnosis.json: complete prediction comparisons, parser audit, validation attention statistics, and counterfactual results.
- ../continuation_s0/{original_s0,attention_s0}/ and ../conditioned_s0/: frozen checkpoints, predictions, configs, and training histories.
- ../MSI_FINAL_REPORT.md: original training summary and provenance.

Raw test examples were not inspected to design further changes, and no model was selected or retrained using this analysis.
