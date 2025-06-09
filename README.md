# Acoustic-Based Deception Detection: A Cross-Lingual Analysis Using Machine Learning

## Project Overview

This is the Coursework of QMUL-BUPT Joint Program CBU5201 Machine Learning Module.

This project tackles the complex challenge of automated deception detection in spoken narratives using machine learning techniques. Diverging from traditional lie detection methods that often rely on physiological responses or linguistic content analysis, this approach focuses exclusively on **vocal acoustic properties** to discern whether a narrated story is truthful or deceptive. The goal is to develop a model that can make this determination without analyzing the semantic content of the speech, thereby aiming for a cross-linguistic applicability.

The problem is framed as a **binary classification task**. Given audio recordings (typically 3-5 minutes long) of individuals narrating stories, the model's objective is to predict one of two outcomes: "True Story" or "Deceptive Story."

## Author

* Shulin Li

## Key Features

* **Purely Acoustic Analysis**: Deception detection is performed solely based on extracted acoustic features, making the approach language-agnostic.
* **Comprehensive Feature Extraction**: Utilizes `librosa` and `pydub` to extract a rich set of acoustic features including:
    * Mel-Frequency Cepstral Coefficients (MFCCs)
    * Chroma Feature
    * Mel Spectrogram
    * Spectral Contrast
    * Tonnetz
    * Root Mean Square (RMS) Energy
    * Zero Crossing Rate
    * Pitch (using `pydub`'s `detect_pitch_frequency`)
    * Jitter and Shimmer (using `pydub`'s voice analysis)
    * Harmonics-to-Noise Ratio (HNR)
* **Machine Learning Classification**: The project involves training and evaluating various machine learning models for binary classification. (Based on your notebook, this would typically involve models like SVM, Random Forest, Logistic Regression, etc., which you'd integrate after feature extraction).
* **Structured Workflow**: The Jupyter Notebook provides a clear, step-by-step workflow from data loading and feature extraction to potential model training and evaluation.

## Dataset

The project relies on a custom dataset, which includes audio recordings of narrated stories along with their corresponding truthfulness labels.

* **Audio Files**: Expected to be located in the `data/CBU0521DD_stories/` directory. These are the raw audio inputs for feature extraction.
* **Metadata**: The `data/CBU0521DD_stories_attributes.csv` file contains crucial metadata, including the `Story_type` (True/Deceptive) for each audio file, which serves as the ground truth.

### Data Download Instructions

The full audio dataset (`CBU0521DD_stories`) can be obtained from Kaggle:

**[CBU0521DD Stories Dataset on Kaggle](https://www.kaggle.com/datasets/rebornxd/cbu0521dd-stories)**

Please download the dataset from the link above. After downloading, extract its contents into the `data/` directory of this project, ensuring the following structure:


Acoustic-Deception-Detection/

└── data/

└── CBU0521DD_stories/

├── audio_file_1.wav

├── audio_file_2.wav

└── ...


## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Shulin-Li22/CBU5201_2425_Acoustic-Deception-Detection.git](https://github.com/Shulin-Li22/CBU5201_2425_Acoustic-Deception-Detection.git)
    cd Acoustic-Deception-Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Project Execution

1.  **Ensure Data Availability**: Make sure you have downloaded and placed the audio files as per the [Data Download Instructions](#data-download-instructions) section.
2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
3.  **Open and Run the Notebook**: In the Jupyter interface, navigate to and open `CBU5201_miniproject_submission.ipynb`. Run all cells in the notebook sequentially. The notebook will:
    * Load the metadata.
    * Extract acoustic features from the audio files.
    * Save the extracted features to `audio_features.csv`.
    * (Optionally) Perform model training, evaluation, and visualization if implemented in the notebook.

## Project Structure

.
├── README.md                                   # Project overview and instructions

├── CBU5201_miniproject_submission.ipynb        # Main Jupyter Notebook with core logic

├── data/                                       # Contains datasets

│   ├── CBU0521DD_stories/                      # Audio files (often downloaded separately)

│   └── CBU0521DD_stories_attributes.csv        # Metadata for audio files

├── src/                                        # (Optional) For modularized code

│   ├── audio_feature_extractor.py              # Refactored feature extraction logic

│   └── model_trainer.py                        # Refactored model training logic

├── requirements.txt                            # List of Python dependencies

├── .gitignore                                  # Specifies intentionally untracked files to ignore

└── LICENSE                                     # Project's license file


## Technologies Used

* Python 3.8+
* Jupyter Notebook
* `librosa`
* `pydub`
* `pandas`
* `numpy`
* `scikit-learn` (for machine learning models)
* `matplotlib` (for visualizations)
* `seaborn` (for enhanced visualizations)


## Future Work

* **Deep Learning Models**: Investigate the applicability of deep learning architectures (e.g., Convolutional Neural Networks for spectrograms, Recurrent Neural Networks/LSTMs for sequential features) for improved deception detection.

* **Expanded Dataset**: Gather or utilize a larger and more diverse dataset, especially one encompassing multiple languages, to robustly test cross-linguistic generalization.

* **Feature Engineering**: Explore additional acoustic or prosodic features that might be indicative of deception (e.g., speech rate, pauses, vocal tension).

* **Real-time Processing**: Develop a module for real-time acoustic feature extraction and prediction for potential live applications.

* **Model Deployment**: Consider deploying the trained model as a web API or a user-friendly application.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

* Special thanks to Queen Mary University of London / CBU5201 Machine Learning for the opportunity to work on this interesting project.
* Thanks to the creators of `librosa` and `pydub` for their excellent audio processing libraries.