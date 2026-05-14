# Signal Peptide CPP

CPP-based interpretable signal peptide prediction using physicochemical feature engineering.

## Current pipeline

- SignalP 6.0 preprocessing
- CPP feature extraction
- Binary SP vs Non-SP classification
- SP type classification using one-vs-rest classifiers
- Pairwise refinement for closely related SP types
- CPP feature interpretation and visualization
- External benchmark end-to-end evaluation

## Repository structure

```text
scripts/
    01_preprocess_signalp6.py

notebooks/
    01_binary_classifier.ipynb
        CPP-based SP vs Non-SP classificatio

    02_sp_type_classifier.ipynb
        SP type classification using OVR classifiers and pairwise refinement

    03_cpp_type_vs_nonsp_interpretation.ipynb
        Biological CPP interpretation of each SP type against Non-SP sequences

    04_benchmark_end_to_end_eval.ipynb
        External benchmark evaluation of the complete prediction pipeline