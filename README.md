
## Standard Sturucture for Coding
```bash

mkdir reports/

mkdir src/
mkdir src/dataset/
mkdir src/experiments/
mkdir src/models/
mkdir src/scripts/
touch src/scripts/trainer.py

```
Format `python trainer.py --dataset <dataset> --model <model>`

---
## END to END - Kaggle Submission Pipeline
end to end flow of a "batch"
  - [x] 1. dataset
    - [x] training & test split as img_folder format
    - [x] testing with assert statements
    - [ ] remaining validation split folder seperation.
  - [ ] 2. model
    - [x] simple model
    - [x] test single batch passing through model
  - [ ] 3. model training
  - [ ] 4. model experimenting
  - [ ] 5. kaggle submission on test data
