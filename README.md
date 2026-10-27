# Gau Pehchan — Cattle and Buffalo breed recognition

A local web app for uploading or taking a livestock photo and getting a visual breed shortlist. The frontend and backend are served by one Python file, `app.py`. It runs on your computer and sends images to the local service, not a hosted inference API.

## Start the app

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and install the app dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the server:

   ```bash
   python app.py
   ```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The upload page and API are both served locally. Press Ctrl+C in the terminal to stop the app. API documentation is at http://127.0.0.1:8000/api/docs.

## Add a trained breed model

The repository does not include trained weights or livestock photos. Predictions become available once you train and install a compatible model. Without one, the interface clearly reports that the model is missing and the prediction endpoint returns an error; it never invents results.

The included training script fine-tunes MobileNetV3 Small. Use this folder layout, with the same breed folders in `train` and `val`:

```
data/
  train/
    cattle__Gir/
    cattle__Sahiwal/
    buffalo__Murrah/
    buffalo__Mehsana/
  val/
    cattle__Gir/
    cattle__Sahiwal/
    buffalo__Murrah/
    buffalo__Mehsana/
```

Put each labeled photo in its class folder. Class names start with `cattle__` or `buffalo__`; use a name such as `cattle__Crossbred_cattle` for crossbreds. Install training packages and run:

```bash
python -m pip install -r training-requirements.txt
python train_model.py --data data --epochs 12
```

Training downloads the standard MobileNetV3 initialization the first time it runs and writes `model.onnx` and `breeds.json`. The app reads those files from the same folder as `app.py`. Restart the app after adding or replacing the files. You can place them elsewhere by setting `BREED_MODEL` and `BREED_LABELS` environment variables.

The model metadata must match the exported model's class order and preprocessing. `breeds.example.json` shows the expected format. The included training script creates the matching metadata automatically.

## Before field use

Use consented, representative images and keep separate training and validation animals where possible. Check per-breed performance across age, sex, lighting, pose and crossbreeding; an overall validation score is not enough. This is a visual estimate, not pedigree verification. A photo cannot calculate coefficient of inbreeding (COI), which requires verified lineage or genetic data. The included app stores saved record labels in the current browser's local storage; it does not provide a shared database or user accounts.

## GitHub

Commit the source, README and dependency files. Keep private or consent-restricted animal photos out of the repository. Model weights are optional; include `model.onnx` and `breeds.json` only if you want a clone of the repository to have the trained model immediately. Do not commit secrets or private farm records.
