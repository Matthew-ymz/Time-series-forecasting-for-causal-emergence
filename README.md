# Machine Learning Lab 

This repository contains time series machine learning implementations from Zhang Jiang's research group, supporting NN and iTransformer models. The code framework is adapted from the following research:

> Wu H, Hu T, Liu Y, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *The Eleventh International Conference on Learning Representations*, 2022.  
> [[Paper]](https://arxiv.org/abs/2210.02186) | [[Official Implementation]](https://github.com/thuml/TimesNet)

For comprehensive benchmarks and technical details, please refer to the original TimesNet publication.

---

## Quick Start

### 1. Installation
Ensure Python 3.11 is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Prepare your dataset according to the required format.

### 3. Model Training & Evaluation
Experiment scripts for benchmarks are provided in `./scripts/`. Example usage:
```bash
# Forecasting example
bash scripts/long_term_forecast/SIR/NN.sh
```

### 4. Custom Model Development
To implement a new model:
1. **Add Model File**: Place your model implementation in `./models/` (reference `./models/Transformer.py`).
2. **Register Model**: Include the model in `Exp_Basic.model_dict` within `./exp/exp_basic.py`.
3. **Create Script**: Add corresponding experiment scripts under `./scripts/`.


## Contact
For inquiries or suggestions, contact:  
**Mingzhe Yang**  
📧 202321250027@mail.bnu.edu.cn


## Acknowledgements
This project builds upon the following open-source repositories:
- [Autoformer (Forecasting)](https://github.com/thuml/Autoformer)
- [Anomaly-Transformer (Anomaly Detection)](https://github.com/thuml/Anomaly-Transformer)
- [Flowformer (Classification)](https://github.com/thuml/Flowformer)
