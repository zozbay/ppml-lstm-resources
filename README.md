# Predictive Process Monitoring with Resources

Reproduction and improvement of Tax et al. (2017) "Predictive Business Process Monitoring with LSTM Neural Networks"

**Authors:** Tsogt Batbaatar, Nourcene Brahem, Zeynep Deniz Özbay  
**Course:** PPML WiSe 2025/26, Humboldt University Berlin  
**GitHub:** [zozbay/ppml-lstm-resources](https://github.com/zozbay/ppml-lstm-resources)

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

**Change Type:** Input features (#2)

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
│   ├── train.py                                      # Helpdesk baseline
│   ├── train_bpic_baseline.py                       # BPIC baseline (no resources)
│   ├── train_bpic_resources.py                      # BPIC with resources (our improvement)
│   ├── evaluate_suffix_and_remaining_time.py        # Helpdesk evaluation
│   ├── evaluate_suffix_and_remaining_time_bpic.py   # BPIC baseline evaluation
│   ├── evaluate_suffix_and_remaining_time_resources.py # Resources evaluation
│   ├── calculate_accuracy_on_next_event.py          # Helpdesk accuracy
│   ├── calculate_accuracy_on_next_event_bpic.py     # BPIC baseline accuracy
│   ├── calculate_accuracy_on_next_event_resources.py # Resources accuracy
│   └── evaluate_next_activity_and_time.py           # Alternative evaluation
├── data/
│   ├── helpdesk.csv                                  # Helpdesk dataset
│   ├── bpi_12_w.csv                                  # BPIC 2012 W (no resources)
│   ├── bpic2012_w_resources.csv                      # BPIC 2012 W with resources
│   ├── bpic2012_with_resources.csv                   # Full BPIC 2012 with resources
│   ├── BPI_Challenge_2012.xes                        # Original XES file (can be downloaded)
│   ├── xes_to_csv.py                                 # XES extraction script
│   └── w_act_filter.py                               # Filter to W_ activities
├── output_files/
│   ├── models/                                       # Trained models (.h5 files)
│   ├── results/                                      # Evaluation results
│   └── folds/                                        # Cross-validation folds
├── .gitignore
└── README.md
```

## Environment Setup

**Requirements:**
- Python 3.11.9
- TensorFlow 2.20.0
- Keras 3.13.0

**Installation:**
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
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

**Expected output:** ~71.25% accuracy

### 2. Reproduce Baseline (BPIC 2012)
```bash
python train_bpic_baseline.py
python evaluate_suffix_and_remaining_time_bpic.py
python calculate_accuracy_on_next_event_bpic.py
```

**Expected output:** ~76.20% accuracy

### 3. Train with Resources (Our Improvement)
```bash
python train_bpic_resources.py
python evaluate_suffix_and_remaining_time_resources.py
python calculate_accuracy_on_next_event_resources.py
```

**Expected output:** ~78-80% accuracy

## Data Processing

## Data Download
Download `BPI_Challenge_2012.xes` from [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)

### Extract Resources from XES
```bash
cd data
python xes_to_csv.py
```

This converts `BPI_Challenge_2012.xes` to `bpic2012_with_resources.csv` with resource information.

### Filter to W_ Activities
```bash
python w_act_filter.py
```

This filters to only W_ activities (manually executed work items) and creates `bpic2012_w_resources.csv`.

## Dataset Information

**BPIC 2012 W Subprocess:**
- **Cases:** 13,087
- **Activities:** 6 W_ activities (manually executed)
- **Events:** 72,414
- **Resources:** 69 unique

**W_ Activities:**
1. `W_Completeren aanvraag` - Complete application
2. `W_Nabellen offertes` - Call regarding offers
3. `W_Valideren aanvraag` - Validate application
4. `W_Nabellen incomplete dossiers` - Call incomplete dossiers
5. `W_Afhandelen leads` - Handle leads
6. `W_Beoordelen fraude` - Assess fraud

**Time Features (calculated from timestamps):**
1. Time since last event
2. Time since case start
3. Time since midnight
4. Day of week
5. Position in trace

## Model Architecture

- **Architecture:** 2-layer LSTM
- **Neurons:** 100 per layer
- **Optimizer:** Nadam (lr=0.002)
- **Early Stopping:** Patience=42
- **Validation Split:** 20%
- **Multi-task:** Next activity + remaining time prediction

## Results Summary

### Baseline Reproduction
- **Helpdesk:** 71.25% accuracy (matches paper's ~71%)
- **BPIC 2012 W:** 76.20% accuracy (matches paper's ~76%)

### With Resources (In Progress)
- **Training:** Complete (val_loss 1.53 vs baseline 1.67)
- **Evaluation:** In progress
- **Expected:** ~78-80% accuracy

## Key Findings

1. **Successful Reproduction:** Both baselines reproduced within 0.25% of original paper
2. **Lower Validation Loss:** Resource model achieved 1.53 vs 1.67 baseline
3. **Feature Expansion:** 11 → 80 features (7x increase)
4. **Implementation:** Same architecture ensures fair comparison

## Files Summary

**Training Scripts:**
- `train.py` - Original Helpdesk baseline
- `train_bpic_baseline.py` - BPIC baseline without resources
- `train_bpic_resources.py` - **Our improvement with resources**

**Evaluation Scripts:**
- `evaluate_suffix_and_remaining_time_*.py` - Generate predictions
- `calculate_accuracy_on_next_event_*.py` - Calculate accuracy metrics

**Data Processing:**
- `xes_to_csv.py` - Extract data from XES format with resources
- `w_act_filter.py` - Filter to W_ activities only

## Models Trained

| Model | Dataset | Features | Val Loss | Accuracy | Status |
|-------|---------|----------|----------|----------|--------|
| `model_17-1.51.h5` | Helpdesk | 14 | 1.51 | 71.25% | completed |
| `model_bpic_29-1.67.h5` | BPIC W | 11 | 1.67 | 76.20% | completed |
| `model_resources_29-1.53.h5` | BPIC W | 80 | 1.53 | upcoming | upcoming |

## Future Work

- Complete evaluation of resource-enhanced model
- Analyze which resources contribute most to prediction accuracy
- Test on other event logs with resource information
- Experiment with resource availability features

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

## Acknowledgments

- Original implementation: [Ilya Verenich](https://github.com/verenich/ProcessSequencePrediction)
- Course: PPML WiSe 2025/26, Humboldt University Berlin
- Instructors: [Priv.-Doz. Dr. Kate Revoredo]

## License

Based on the original work by Tax et al. (2017). See original repository for license details.