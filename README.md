# On the Expected Size of Conformal Prediction Sets

This repository contains the code for the paper:

[Guneet Singh Dhillon](https://guneet-dhillon.github.io/), [George Deligiannidis](https://www.stats.ox.ac.uk/~deligian/), [Tom Rainforth](https://www.robots.ox.ac.uk/~twgr/)  
**On the Expected Size of Conformal Prediction Sets** ([pdf](https://arxiv.org/pdf/2306.07254.pdf))  
*In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), 2024*

## Abstract

While conformal predictors reap the benefits of rigorous statistical guarantees on their error frequency, the size of their corresponding prediction sets is critical to their practical utility. Unfortunately, there is currently a lack of finite-sample analysis and guarantees for their prediction set sizes. To address this shortfall, we theoretically quantify the expected size of the prediction sets under the split conformal prediction framework. As this precise formulation cannot usually be calculated directly, we further derive point estimates and high-probability interval bounds that can be empirically computed, providing a practical method for characterizing the expected set size. We corroborate the efficacy of our results with experiments on real-world datasets for both regression and classification problems.

## Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{dhillon2024expected,
  title={On the Expected Size of Conformal Prediction Sets},
  author={Dhillon, Guneet S. and Deligiannidis, George and Rainforth, Tom},
  booktitle={Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages={1549--1557},
  year={2024},
  editor={Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume={238},
  series={Proceedings of Machine Learning Research},
  month={02--04 May},
  publisher={PMLR},
  pdf={https://proceedings.mlr.press/v238/dhillon24a/dhillon24a.pdf},
  url={https://proceedings.mlr.press/v238/dhillon24a.html}
}
```

## Usage

### Dependencies

Use Python version 3.9. To download the dependencies, run
```
pip install -r requirements.txt
```

### Computing the expected conformal prediction set size

To compute the expected conformal prediction set size, run
```
python main_run.py --type $type --alpha $alpha --gamma $gamma --frac_train $frac_train --frac_cal $frac_cal --it_train $it_train --it_cal $it_cal
```
with the following arguments:
- type          : Conformal predictor type ('L1Regression', 'ZeroOneClassification', 'CQRRegression', 'LACClassification', 'APSClassification', 'L1HighDimensionRegression', 'L2HighDimensionRegression')
- alpha         : Conformal predictor significance level (default=0.1)
- gamma         : Prediction set size interval significance level (default=0.1)
- frac_train    : Fraction used as training dataset (default=0.25)
- frac_cal      : Fraction used as calibration dataset (default=0.25)
- it_train      : Number of iterations for sampling a new training dataset (default=10)
- it_cal        : Number of iterations for sampling a new calibration dataset (default=100)

To analyze and print the computed results, run
```
python main_analyze.py
```
