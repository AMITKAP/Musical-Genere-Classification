# Music Genre Classification with Deep Learning

Automatically classify songs into genres using audio features and machine learning models. This project uses the GTZAN Music Genre Dataset and demonstrates feature extraction, model training, and evaluation for music genre prediction.

## Dataset Used: GTZAN Music Genre Dataset
- **Total Tracks:** 1000 audio files
- **Duration:** 30 seconds per file
- **Format:** .wav
- **Genres:** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

## Project Workflow
1. **Audio Preprocessing**
    - Normalize audio files
    - Remove noise/silence
2. **Feature Extraction**
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Mel Spectrograms
    - Chroma Features
    - Spectral Centroid, Bandwidth, Zero-Crossing Rate
3. **Model Development**
    - XGBoost for genre classification
    - KNN for music recommendation
4. **Evaluation**
    - Accuracy, confusion matrix, precision, F1-score
    - Train/test split and cross-validation

## Getting Started
1. Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```
2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the notebook:
    Open `ML_to_classify_musical_genre.ipynb` in Jupyter or VS Code and follow the steps.

## Requirements
See `requirements.txt` for all required Python packages.

## Results



## Streamlit UI: streamlit_song_selector.py (Final)

This is the final interactive web app for music genre recommendation and audio preview.

- Select a song from a dropdown (only 'hiphop' and 'rock' genres).
- Preview the selected song with an audio player.
- Get recommendations for similar songs from the dataset, with audio previews for each recommended track.
- Uses KNN for recommendations and StandardScaler for feature scaling.
- Only 'hiphop' and 'rock' genres are supported in the UI; other genres are filtered out.

### How to Run the App
1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Start the Streamlit app:
    ```bash
    cd ui
    streamlit run streamlit_song_selector.py
    ```
3. Open the provided local URL in your browser.

### Notes
- Ensure `Data/features_3_sec.csv` and audio files in `Data/genres_original/` are present.
- Only files from 'hiphop' and 'rock' genres will be recommended and previewed.

## License
This project is for educational purposes. Please check dataset and code licenses before commercial use.

## Acknowledgments
- GTZAN Music Genre Dataset
- Librosa, scikit-learn, XGBoost, TensorFlow
