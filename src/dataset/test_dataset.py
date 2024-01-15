
import tensorflow as tf

dataset_path = "/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/kaggle_dog_breed/src/dataset/dog-breed-imagefolder"
training_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path, 
                                            image_size = (224, 224),  # Default value is (256, 256). **source of potential error**
                                            validation_split = 0.1, subset = "both", seed = 10, # Need to set subset & seed both for validation_split
                                            batch_size = 32, shuffle= True, # Default values used automatically. **SOURCE OF POTENTIAL ERROR**
                                            color_mode = "rgb") # Channels = 3. Hidden from us

#%%
print(training_dataset, validation_dataset)

single_batch    = next(iter(training_dataset))
image           = single_batch[0]
label           = single_batch[1]

assert len(image.shape) == len("BCHW")      # Format is BHWC for Tensorflow & Keras. & BCHW for Pytorch
assert image.shape[0]   == 32               # B
assert image.shape[1:3] == (224,224)        # Height & Width
assert image.shape[3]   == 3                # Channels

try:
    training_dataset[0]                     # 0th Index Example
    training_dataset.take(10)[0][0:2]       # 10 Examples, (Tuple = Image, Label)
    training_dataset.batch(2)[0][0:2]       # 2 Batches, with Batch size 32
except Exception as inst:
    print("This try batch block has errors. When trying to subscript")
