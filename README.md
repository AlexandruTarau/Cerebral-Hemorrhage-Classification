# Cerebral Hemorrhage Classification in CT Scans

This project implements an end-to-end Deep Learning pipeline to classify six types of cerebral hemorrhages (Epidural, Intraparenchymal, Intraventricular, Subarachnoid, Subdural, and "Any") using the **RSNA Intracranial Hemorrhage Detection** dataset.

The system utilizes a fine-tuned **EfficientNet-B0** architecture, optimized for multi-label medical diagnostics through advanced loss functions and medical-specific data augmentations.

## 🚀 Key Features
* **Architecture**: EfficientNet-B0 backbone (pre-trained on ImageNet) with a modified input layer for single-channel (grayscale) CT scans.
* **Loss Function**: Utilizes **Focal Loss** ($\alpha=0.5, \gamma=2.0$) to address extreme class imbalance by down-weighting easy samples and focusing on difficult minority classes.
* **Training Strategy**: 5-fold Stratified Cross-Validation to ensure model stability and reliable performance estimation.
* **Advanced Scheduling**: Implements `ReduceLROnPlateau` to reduce the learning rate by a factor of 0.1 when validation loss stagnates.
* **Explainability**: Integrated **Grad-CAM** visualizations to analyze the model's decision-making process by highlighting relevant anatomical regions.



## 📊 Performance & Results
The model demonstrated strong generalization capabilities, achieving high AUC and stable metrics across all folds.

| Metric | Validation (Mean ± Std) | Test (Mean ± Std) |
| :--- | :--- | :--- |
| **AUC** | **~0.91** | **~0.91** |
| **Accuracy** | $0.6449 \pm 0.0192$ | $0.2548 \pm 0.0159$ |
| **Precision** | $0.6960 \pm 0.0341$ | $0.6781 \pm 0.0165$ |
| **Recall** | $0.4633 \pm 0.0552$ | $0.4698 \pm 0.0560$ |
| **F1-Score** | $0.5360 \pm 0.0342$ | $0.5322 \pm 0.0343$ |

> **Note**: A custom classification threshold of **0.42** was utilized to optimize the balance between Precision and Recall.

## 📂 Project Structure
```text
├── main.py              # Orchestrates the 5-fold training and evaluation loop
├── Trainer.py           # Core logic for training, Focal Loss, and evaluation metrics
├── Preprocess.py        # Custom CV2-based pipeline (CLAHE, Sharpening, Blur)
├── MonaiPreprocess.py   # Advanced MONAI-based augmentations (Rand2DElastic)
├── RSNADataset.py       # Custom PyTorch Dataset for RSNA PNG frames
├── config.json          # Hyperparameters, paths, and transformation toggles
└── Bonus1.py            # Script for generating Grad-CAM heatmaps
