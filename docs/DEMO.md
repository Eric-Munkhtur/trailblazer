# Verdia judging demo

## About one minute

1. Open [the live site](https://verdia-hackru-20839887.westus2.cloudapp.azure.com/) with an empty search. The graph is off.
2. Search **NCT01132482** or click **Sildenafil**. Explain: “We fetch the study from ClinicalTrials.gov and map its drug and disease into PrimeKG.”
3. As nodes and edges illuminate: “These are graph relationships reached by the search. The 3D coordinates are a visualization layout.”
4. Open the prediction panel: “Our trained model reads molecular structure, disease codes, and eligibility sentences. The graph explorer is a separate context view.”
5. Show our score beside the matched original HINT score. “We replaced equal eligibility-sentence averaging with learned attention. Our controlled experiments did not establish an improvement, and we report all results.”
6. Show input provenance / download the run record, then the repository's experiment table and paper.

Backup supported example: **NCT01928927**. A model score such as 0.657 is not 65.7% accuracy. Historical trials demonstrate the application; this is not prospective clinical validation.

## Unsupported study example

**NCT02830178** is observational, without a Phase II drug intervention. The graph can map its conditions, but the page explains why the Phase II outcome model does not score it. Do not promise scores for every NCT ID.

## If the network or model is busy

Use the captured screenshots in `deploy/evidence/` and clearly identify them as recorded demonstrations. They are not an offline inference mode. The hosted app currently allows two concurrent graph requests and a serial prediction queue; venue Wi-Fi can share the per-IP allowance. Have judges try it one at a time during the presentation.
