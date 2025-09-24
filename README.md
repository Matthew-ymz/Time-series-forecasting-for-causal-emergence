# Machine Learning Lab 

This repository contains time series machine learning implementations from Zhang Jiang's research group, supporting NN and iTransformer models. The code framework is adapted from the following research:

> Wu H, Hu T, Liu Y, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *The Eleventh International Conference on Learning Representations*, 2022.  
> [[Paper]](https://arxiv.org/abs/2210.02186) | [[Official Implementation]](https://github.com/thuml/Time-Series-Library)

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

For simulated data, run the corresponding data generation code in the preprocessing folder.  Example usage:

```bash
python preprocessing/couzin_model_python-5.py 
```

### 3. Model Training & Evaluation
Experiment scripts for benchmarks are provided in `./scripts/`. Example usage:
```bash
# Forecasting example
bash scripts/SIR/MLP.sh
```

### 4. Custom Model Development
To implement a new model:
1. **Add Model File**: Place your model implementation in `./models/` (reference `./models/Transformer.py`).
2. **Register Model**: Include the model in `Exp_Basic.model_dict` within `./exp/exp_basic.py`.
3. **Create Script**: Add corresponding experiment scripts under `./scripts/`.


### 5. Dynamical Analysis
Use the postprocessing folder to analyze the singular value spectrum, eigenvectors, and long-term evolution trajectories of the trained dynamics. The corresponding tools are available in `./postprocessing/svd_tool.py`

## Contact
For inquiries or suggestions, contact:  
**Mingzhe Yang**  
📧 202321250027@mail.bnu.edu.cn


## Acknowledgements
This project builds upon the following open-source repositories:
- [Autoformer (Forecasting)](https://github.com/thuml/Autoformer)
- [Anomaly-Transformer (Anomaly Detection)](https://github.com/thuml/Anomaly-Transformer)
- [Flowformer (Classification)](https://github.com/thuml/Flowformer)
