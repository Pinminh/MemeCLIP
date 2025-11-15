# MemeCLIP for ViHarMe: Harmful Meme Detection in Vietnamese Media

## About MemeCLIP

MemeCLIP is a multimodal framework designed for efficient meme classification that preserves the knowledge of pre-trained CLIP models while adapting to downstream tasks. Originally proposed by Shah et al. (2024) and published at EMNLP 2024, MemeCLIP was developed to address multiple aspects of meme analysis including hate speech detection, target identification, stance classification, and humor detection.

The framework leverages CLIP's powerful vision-language representations and introduces several key innovations:

- **Feature Adapters**: Lightweight adapter modules that enable efficient fine-tuning while preventing overfitting on small datasets
- **Dual Projection Layers**: Separate mapping layers for image and text embeddings that preserve multimodal information
- **Cosine Classifier**: A robust classification head that handles imbalanced data effectively
- **Residual Connections**: Combines adapter outputs with original projections to maintain pre-trained knowledge

MemeCLIP outperforms baseline CLIP models and other multimodal methods across various meme classification tasks, demonstrating particular robustness in handling class imbalance and limited training data.

<p align="center">
  <img src="MemeCLIP.png" />
</p>

This is a modified repository of the original paper [MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification](https://aclanthology.org/2024.emnlp-main.959/) published in [EMNLP 2024](https://2024.emnlp.org/), whose repository is available at [here](https://github.com/SiddhantBikram/MemeCLIP).

## Adaptation to ViHarMe Dataset

This repository contains a modified implementation of MemeCLIP adapted for the **ViHarMe** (Vietnamese Harmful Meme) dataset - a specialized dataset for detecting harmful memes in Vietnamese social media contexts.

### ViHarMe Dataset Structure

Each sample in the ViHarMe dataset contains:

```json
{
    "image": "image filename",
    "topic": "the topic of meme",
    "label": "harmful" or "harmless",
    "target": "the target of the harmfulness",
    "type": "the type or reason making it harmful"
}
```

### Key Adaptations

1. **Binary Classification Setup**: The model is configured for binary classification (harmful vs. harmless) to address the primary task of identifying harmful Vietnamese memes.

2. **Multi-Target Classification Support**: The code also supports target classification with five categories:
   - Individual
   - Organization
   - Community
   - Local
   - Society

3. **Vietnamese Text Processing**: The implementation handles Vietnamese text through CLIP's multilingual tokenization capabilities, processing meme captions in Vietnamese language.

4. **Dataset-Specific Configuration**: Modified `configs.py` to work with ViHarMe's CSV format and directory structure:
   ```python
   cfg.root_dir = '/content/drive/MyDrive/ViHarMe/sample-data'
   cfg.img_folder = '/content/drive/MyDrive/ViHarMe/sample-data/images'
   cfg.info_file = '/content/drive/MyDrive/ViHarMe/sample-data/memeclip.csv'
   ```

5. **Custom Label Mapping**: Adapted the collator to handle ViHarMe's "harmful"/"harmless" labels instead of the original PrideMM annotation scheme.

## Experimental Setup

### Model Configuration

- **CLIP Variant**: ViT-L/14 (Large Vision Transformer)
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Maximum Epochs**: 10
- **Adapter Ratio**: 0.2 (balancing adapted and original features)
- **Feature Dimensions**:
  - Unmapped dimension: 768
  - Mapped dimension: 1024

### Training Pipeline

1. **Feature Extraction**: CLIP encoders extract image and text features without the projection layer
2. **Dual Mapping**: Separate linear projections map features to a common 1024-dimensional space
3. **Adaptation**: Feature adapters with reduction ratio of 4 refine the representations
4. **Fusion**: Element-wise multiplication combines image and text features
5. **Classification**: A cosine classifier with temperature scaling (scale=30) produces final predictions

### Metrics

The model is evaluated using:
- **Accuracy**: Overall classification correctness
- **AUROC**: Area under the ROC curve for binary classification performance
- **Precision, Recall, F1**: Weighted metrics to account for potential class imbalance

## Usage

### Data Preparation

Prepare your CSV file with the following columns:
- `name`: Image filename
- `text`: Meme text/caption (use "None" for memes without text)
- `label`: "harmful" or "harmless"
- `target`: Target category (for target classification task)
- `split`: "train", "val", or "test"

### Configuration

All experimental settings can be modified in `configs.py`:

```python
# Task selection
cfg.label = 'label'  # Use 'label' for harm detection, 'target' for target classification

# Class names
cfg.class_names = ['harmless', 'harmful']  # For harm detection
# cfg.class_names = ['individual', 'organization', 'community', 'local', 'society']  # For target classification

# Training mode
cfg.test_only = False  # Set to True for testing only
```

### Training

To train MemeCLIP on ViHarMe:

```bash
python main.py
```

The training process will:
1. Load the ViHarMe dataset splits
2. Extract CLIP features for all samples
3. Train the model with automatic validation
4. Save the best checkpoint based on validation AUROC
5. Evaluate on the test set using the best model

### Testing

To test a pre-trained model:

1. Set `cfg.test_only = True` in `configs.py`
2. Specify the checkpoint path in `cfg.checkpoint_file`
3. Run `python main.py`

## Implementation Details

### Custom Components

- **Custom_Dataset**: Handles ViHarMe data loading with support for missing text values
- **Custom_Collator**: Extracts CLIP features on-the-fly and prepares batches
- **Feature Adapters**: Lightweight modules (c_in → c_in//4 → c_in) for efficient adaptation
- **Cosine Classifier**: Temperature-scaled cosine similarity classification

### Key Modifications from Original MemeCLIP

1. **Label Processing**: Binary "harmful"/"harmless" labels replace multi-class hate/humor labels
2. **Vietnamese Support**: Native handling of Vietnamese text through CLIP's multilingual capabilities
3. **Target-Specific Filtering**: For target classification, only "harmful" samples are included in training
4. **Simplified Pipeline**: Streamlined for Google Colab/Drive integration

## Results

The model evaluates performance on the ViHarMe test set using:
- Classification accuracy
- AUROC for discriminative capability
- Weighted precision, recall, and F1-score

Results are logged during training and testing phases for comprehensive performance analysis.

## Requirements

- PyTorch
- PyTorch Lightning
- CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- torchmetrics
- transformers
- pandas
- Pillow

## Citation

If you use this adapted implementation for ViHarMe experiments, please cite the original MemeCLIP paper:

```bibtex
@inproceedings{shah2024memeclip,
    title = "MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification",
    author = "Shah, Siddhant Bikram  and
      Shiwakoti, Shuvam  and
      Chaudhary, Maheep  and
      Wang, Haohan",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.959/",
    doi = "10.18653/v1/2024.emnlp-main.959",
    pages = "17320--17332",
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the original MemeCLIP framework by Shah et al. (2024), adapted specifically for Vietnamese harmful meme detection research. The ViHarMe dataset provides crucial resources for understanding harmful content in Vietnamese social media contexts.

---

**Note**: The ViHarMe dataset contains sensitive content related to harmful memes. Please use this resource responsibly and only for research purposes aimed at improving online safety and content moderation.
