# Predictive Process Monitoring with Resources

Reproduction and improvement of Tax et al. (2017) "Predictive Business Process Monitoring with LSTM Neural Networks"

**Authors:** Tsogt Batbaatar, Nourcene Brahem, Zeynep Deniz Oezbay  
**Course:** PPML WiSe 2025/26, Humboldt University Berlin

## Overview

This project reproduces the LSTM-based predictive process monitoring approach from Tax et al. (2017) and extends it by incorporating resource information as additional features.

## Original Paper

- **Title:** Predictive Process Monitoring with LSTM Neural Networks
- **Authors:** Niek Tax, Ilya Verenich, Marcello La Rosa, Marlon Dumas
- **Conference:** CAiSE 2017
- **DOI:** [10.1007/978-3-319-59536-8_30](https://doi.org/10.1007/978-3-319-59536-8_30)
- **Original Repository:** [verenich/ProcessSequencePrediction](https://github.com/verenich/ProcessSequencePrediction)

## Reproduction Results

| Dataset | Metric | Original Paper | Our Result | Status |
|---------|--------|----------------|------------|--------|
| Helpdesk | Next activity accuracy | ~71% | **71.25%** | succesfully reproduced |
| BPIC 2012 W | Next activity accuracy | ~76% | **76.20%** | succesfully reproduced |

## Proposed Improvement

**Change Type:** Input features (Type #2)

**Motivation:** The authors suggested as future work to "extend feature vectors with additional case and event attributes (e.g. resources)" (Tax et al., 2017, p. 14).

**Implementation:**
- Extracted resource information from original BPIC 2012 XES file
- Added 69 one-hot encoded resource features
- **Original:** 11 features (6 activities + 5 time features)
- **Enhanced:** 80 features (6 activities + 5 time features + 69 resources)

## Repository Structure
```
ProcessSequencePrediction/
├── code/
│   ├── train.py                                    # Helpdesk baseline
│   ├── train_bpic_baseline.py                     # BPIC baseline
│   ├── train_with_resources.py                    # BPIC with resources (our improvement)
│   ├── evaluate_suffix_and_remaining_time.py      # Helpdesk evaluation
│   ├── evaluate_suffix_and_remaining_time_bpic.py # BPIC baseline evaluation
│   ├── evaluate_suffix_and_remaining_time_resources.py # Resources evaluation
│   ├── calculate_accuracy_on_next_event.py        # Helpdesk accuracy
│   ├── calculate_accuracy_on_next_event_bpic.py  # BPIC baseline accuracy
│   └── calculate_accuracy_on_next_event_resources.py # Resources accuracy
├── data/
│   ├── helpdesk.csv                               # Helpdesk dataset
│   ├── bpi_12_w.csv                               # BPIC 2012 W subprocess (no resources)
│   ├── bpic2012_w_resources.csv                   # BPIC 2012 W with resources
│   ├── xes_to_csv_with_resources.py              # XES extraction script
│   └── filter_to_w_activities.py                  # W_ activity filter
├── .gitignore
└── README.md
```

## Environment Setup
```bash
# Python version
Python 3.11.9

# Dependencies
pip install tensorflow==2.20.0
pip install keras==3.13.0
pip install numpy pandas scikit-learn
pip install distance jellyfish
```

## How to Run

### 1. Reproduce Baseline (Helpdesk)
```bash
cd code
python train.py
python evaluate_suffix_and_remaining_time.py
python calculate_accuracy_on_next_event.py
```

### 2. Reproduce Baseline (BPIC 2012)
```bash
python train_bpic_baseline.py
python evaluate_suffix_and_remaining_time_bpic.py
python calculate_accuracy_on_next_event_bpic.py
```

### 3. Train with Resources (Our Improvement)
```bash
python train_with_resources.py
python evaluate_suffix_and_remaining_time_resources.py
python calculate_accuracy_on_next_event_resources.py
```

## Results

**Baseline Results:**
- Helpdesk: 71.25% accuracy ✅
- BPIC 2012 W: 76.20% accuracy ✅

**With Resources (Expected):**
- BPIC 2012 W: ~78-80% accuracy (training complete, evaluation in progress)

## Key Files

**Data Processing:**
- `xes_to_csv_with_resources.py` - Extracts resource information from XES
- `filter_to_w_activities.py` - Filters to W_ activities only

**Training:**
- `train_with_resources.py` - Modified to include 69 resource features

**Evaluation:**
- `evaluate_suffix_and_remaining_time_resources.py` - Evaluates model with resources
- `calculate_accuracy_on_next_event_resources.py` - Calculates final accuracy

## Dataset Information

**BPIC 2012 W Subprocess:**
- 13,087 cases
- 6 W_ activities (manually executed work items)
- 72,414 events
- 69 unique resources

**W_ Activities:**
1. W_Completeren aanvraag (Complete application)
2. W_Nabellen offertes (Call regarding offers)
3. W_Valideren aanvraag (Validate application)
4. W_Nabellen incomplete dossiers (Call regarding incomplete dossiers)
5. W_Afhandelen leads (Handle leads)
6. W_Beoordelen fraude (Assess fraud)

## Model Architecture

- 2-layer LSTM
- 100 neurons per layer
- Early stopping (patience=42)
- Validation split: 20%
- Multi-task learning (next activity + remaining time)

## License

This project is based on the original work by Tax et al. (2017). Original repository: https://github.com/verenich/ProcessSequencePrediction

## Citation

If you use this work, please cite the original paper:
```bibtex
@inproceedings{tax2017predictive,
  title={Predictive business process monitoring with LSTM neural networks},
  author={Tax, Niek and Verenich, Ilya and La Rosa, Marcello and Dumas, Marlon},
  booktitle={International Conference on Advanced Information Systems Engineering},
  pages={477--492},
  year={2017},
  organization={Springer}
}
```