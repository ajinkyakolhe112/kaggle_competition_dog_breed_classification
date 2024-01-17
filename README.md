
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

## END to END - Kaggle Submission Pipeline ![](https://geps.dev/progress/60)
**Checklist**: End to end flow of a "batch"
  - [x] 1. dataset
    - [x] training & test split as img_folder format
    - [x] testing with assert statements
    - [ ] remaining validation split folder seperation.
  - [x] 2. model
    - [x] simple model
    - [x] test single batch passing through model
  - [x] 3. model training
    - [x] forward pass
    - [x] error calculation
    - [x] gradient update wrt loss
    - [ ] **reducing of error**
  - [ ] 4. model experimenting
    - [x] model from scratch
    - [x] transfer learning model
    - [ ] weights & bias - monitoring experiments
  - [x] 5. kaggle submission on test data

**IMP: 4 Bugs in the code**
1. `dir = dog-breed-imagefolder`. It lead to 2 classes 1st class -> train & 2nd class -> test, instead of 120 classes as dog breeds
2. default value of `label_mode=int`. it doesn't convert class to vector. categorical loss function needs vector of class not int. `tf.keras.losses.categorical_cross_entropy` vs `tf.keras.losses.sparse_categorical_cross_entropy`. I needed to write custom training loop to debug this problem.
3. `batch_size = 32, shuffle= True`. These default values can lead to confusion while debugging with single element batch.
4. `validation_split` needs subset & seed. Tensorflow requirements
