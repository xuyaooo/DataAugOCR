# An Effective Data Augmentation Method by Asking Questions about Scene Text Images 

Official Implementation for IEEE ICASSP 2026 paper "An Effective Data Augmentation Method by Asking Questions about Scene Text Images"
---

## Environment

- **Python:** 3.9.19
- **PyTorch (CUDA 11.8):**
  - `torch==2.1.0+cu118`
  - `torchaudio==2.1.0+cu118`
  - `torchvision==0.16.0+cu118`
---

## Datasets

### 1. WordArt

- **Source:** [WordArt (ECCV 2022)](https://github.com/xdxie/WordArt) — artistic text recognition (CornerTransformer, MMOCR).
- **Download:** Run the download script from the **WordArt** folder:
  - **TrOCR:** `python TrOCR/WordArt/download_wordart.py`
  - **TrOCRAug:** `python TrOCRAug/WordArt/download_wordart.py`
- The script fetches the dataset from Google Drive and extracts it into the corresponding `WordArt` directory (`train` / `test` images and label files).

### 2. Esposalles

- **Source:** [RRC Challenge 10 – Historical Handwritten Marriage Records](https://rrc.cvc.uab.es/?ch=10&com=introduction). More information can be found [here](https://ieeexplore.ieee.org/document/8270158)
- **Setup:** Use the **Esposalles** layout under `HandTrOCR` and `HandTrOCRAug`:

  **Training:**
  - Download all **3 parts** of the training set from the RRC website.
  - Create a folder named `train`.
  - Put the contents of all 3 parts into this `train` folder (so that each record folder lies directly under `train/`).

  **Test:**
  - Download the **test set** from the RRC website.
  - Take the **Records** (record folders) and place them in a folder named `test`.
  - Inside `test`, create a subfolder named `gt`.
  - Put the ground-truth **XML** files (also from the website) into `test/gt`.

  The resulting structure should look like:

  ```
  HandTrOCR/Esposalles/   (or HandTrOCRAug/Esposalles/)
  ├── train/
  │   ├── <record_folder_1>/
  │   ├── <record_folder_2>/
  │   └── ...
  └── test/
      ├── gt/
      │   ├── <gt_xml_1>.xml
      │   └── ...
      ├── <record_folder_1>/
      ├── <record_folder_2>/
      └── ...
  ```

  Repeat the same `Esposalles` layout for both `HandTrOCR` and `HandTrOCRAug` if you use both.

---

## Training

Train from the project root (or adjust paths as needed):

- **WordArt (TrOCR / TrOCRAug):**  
  `python TrOCR/train_WD.py` or `python TrOCR/train_Straug.py`  
  `python TrOCRAug/train_add.py`
- **Esposalles (HandTrOCR / HandTrOCRAug):**  
  `python HandTrOCR/train_WD.py` or `python HandTrOCR/train_Straug.py`  
  `python HandTrOCRAug/train_add.py`

Each folder contains its own train script; run the one that matches the dataset and variant you want.

---

## Project layout

- **TrOCR / TrOCRAug:** WordArt dataset, TrOCR-based training and augmentation.
- **HandTrOCR / HandTrOCRAug:** Esposalles dataset, handwritten TrOCR training and augmentation.
