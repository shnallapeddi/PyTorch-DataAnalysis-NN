**Title**: Depicting the Crime Incidents(using PyTorch, Neural networks) of Buffalo, New York.

Understanding the pattern of crimes informs safer communities and decision-
making processes. This report examines trends, statistical correlations, and points
of possible intervention using a unique dataset of incidents of crime reported in
Buffalo, New York.

**Source of the data**: https://data.buffalony.gov/Public-Safety/Crime-Incidents/d6g9-xbgu/about_data

Repo has a notebook suite for end-to-end tabular analysis and neural-network experimentation in PyTorch. The repo couples classical EDA/baselines with PyTorch implementations of MLPs for tabular data and a compact CNN for OCTMNIST image classification. Notebooks are self-contained and runnable top-to-bottom.

#### Technical highlights
##### Tabular pipeline (crime incidents)
1. EDA & cleaning: schema inspection, missing-value handling, categorical encoding, outlier checks; temporal slicing for leakage-free splits.
2. Feature engineering: count/frequency encodings, time features (hour/day/weekday), and simple spatial proxies if coordinates exist.
3. Baselines: logistic/linear models or tree baselines for quick calibration (ensures NN gains are meaningful).
4. PyTorch MLP: Dataset / DataLoader abstractions, batch-norm + dropout regularization, non-linear head, BCE/CE losses, early stopping by validation metric.
5. Evaluation: accuracy, precision/recall/F1; calibration/threshold tuning where classification applies.

##### Vision pipeline (OCTMNIST)
1. Model: small CNN with Conv–BN–ReLU blocks, spatial downsampling, global pooling, and fully-connected head.
2. Training loop: SGD/Adam, cosine or step LR schedule, optional gradient clipping; mixed precision if available.
3. Augmentation: lightweight transforms (resize/normalize; optional flips/rotations depending on notebook cell toggles).
4. Metrics: top-1 accuracy and loss curves; confusion matrix for error analysis.

##### Data setup
1. Crime incidents (tabular): download from Buffalo Open Data (link in repo README) and set the CSV path in the first cell of the analysis notebook. 
2. OCTMNIST (vision): run the download/prep cell in the OCTMNIST notebook or point to a local folder; ensure class folders are in the expected layout.
