# PIBI
Standard models that only look at sensor data (macro-observations) are often flying blind. This repository contains the official PyTorch implementation for our paper:

"Uncovering Hidden Degeneration: A Physics-Guided Bidirectional Inference Framework for Industrial Time Series Prediction." We introduce PIBI, a novel framework that bridges the gap between the seen and the unseen. 

It uniquely combines the principles of physics with probabilistic inference to detect hidden degeneration before it becomes a critical problem.

🚀 Key Features See the Unseeable:

PIBI explicitly models unobservable micro-level damage, allowing for much earlier fault prediction than traditional methods. Physics-Meets-Data: We fuse a bottom-up Continuum Damage Mechanics (CDM) simulator with a top-down Maximum Entropy Inference module. This creates a bidirectional loop where physics informs data and data calibrates physics. Robust & Reliable: Our approach delivers more accurate and reliable warnings, even with sparse sensor data, by grounding predictions in physical laws. Real-World Proven: Demonstrated state-of-the-art performance on two challenging, real-world datasets from railway and bridge infrastructure monitoring.

⚙️How It Works: 

The Bidirectional Loop Traditional methods are a one-way street: they look at sensor data and try to guess what's next. PIBI creates a two-way conversation. Macro-to-Micro (Top-Down Inference): Given sensor readings (e.g., track gauge, vibration), our Maximum Entropy module estimates the probability distribution of the hidden micro-damage states. It asks: "Given what we can see, what is the most likely underlying damage?" Micro-to-Macro (Bottom-Up Simulation): We then feed these estimated micro-states into a physics-based CDM simulator. This simulator, governed by the laws of material science, projects how the damage will evolve over time under environmental stressors (like temperature and load). It answers: "If the hidden damage is this, what will the sensors read tomorrow?" This cycle closes a bidirectional loop, allowing the model to continuously refine its understanding of the system's true health and make predictions that are both data-consistent and physically plausible.

🏁 Getting Started

Prerequisites

Python 3.9+ PyTorch 2.0+ NumPy Pandas scikit-learn

🏃‍♀️ Running the Code 

The repository is structured to make it easy to replicate our results and test the model on your own data. Data Preparation The data/ directory contains sample data structured similarly to the datasets used in our paper. The data_processing.py script handles loading and preparing the data for the model. The core model logic is in main_model.py. You can start training the model with a single command. Evaluation The model will automatically evaluate performance on the test set after training is complete, reporting the eight metrics from the paper (F1-score, MCC, ROC-AUC, etc.).

📊 Datasets 

We provide sample data to get you started right away! Heavy-Haul Railway Dataset: A long-term dataset capturing track geometry, load, and environmental variables. KW51 Bridge Dataset: High-frequency vibration and strain data from a monitored bridge. If you want to use your own custom dataset, please format your CSV files with columns for structural indicators, environmental variables, and a target label for failure. You may need to adjust the data_processing.py script to accommodate your specific data structure.

🤝 Acknowledgements

We would like to thank the providers of the railway and bridge datasets for making this research possible.

The complete code will be uploaded gradually, please stay tuned!
