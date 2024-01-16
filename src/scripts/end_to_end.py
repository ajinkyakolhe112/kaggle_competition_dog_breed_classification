
#%%
import tensorflow as tf
from tensorflow import keras

train_dataset_path = "../dataset/dog-breed-imagefolder/train"
training_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(train_dataset_path, 
                                            image_size = (224, 224),  # Default value is (256, 256). **source of potential error**
                                            validation_split = 0.1, subset = "both", seed = 10, # Need to set subset & seed both for validation_split
                                            batch_size = 1, shuffle= True, # Default values used automatically. **SOURCE OF POTENTIAL ERROR**
                                            color_mode = "rgb", # Channels = 3. Hidden from us
                                            label_mode = "categorical" ) # MUST. OTHERWISE causes error at LOSS value calculation. Need to do one hot encoding there
                                            # If you want to provide labels as integers, please use SparseCategoricalCrossentropy loss.
"""
    !IMP: 4 Bugs in the code
    1. for directory, gave path. dog-breed-imagefolder. It lead to 2 classes train & test, instead of 120 classes as dog breeds
    2. label_mode=int. it doesn't convert class to vector. categorical loss function needs vector of class not int. categorical_cross_entropy vs sparse_categorical_cross_entropy. needed to write custom training loop because of this reason.
    3. validation_split needs subset & seed. Tensorflow requirements
    4. default batch_size is 32. shuffle= True. These default values can lead to confusion while debugging with single element batch.
    
"""
#%%
NUM_CLASSES = 120

class simple_fcnn(keras.Model):
    def __init__(self):
        super().__init__()

        self.internal_model = tf.keras.models.Sequential([
            keras.layers.Input(shape = (224, 224, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(units = 240,          activation="relu", ),
            keras.layers.Dense(units = NUM_CLASSES , activation="softmax"),
        ])

    def call(self, input_single_batch):
        final_layer_output = self.internal_model(input_single_batch)

        return final_layer_output

model  = simple_fcnn()
model.compile(
    loss      = keras.losses.CategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam() 
)

x, y = next(iter(training_dataset))
y_pred_probs = model(x)
print(tf.math.reduce_sum(y_pred_probs, axis=1))
#%%
model.fit(x = training_dataset, validation_data = validation_dataset)
#%%
class Trainer():
    def __init__(self):
        self.optimizer          = keras.optimizers.Adam()
        self.loss               = keras.losses.CategoricalCrossentropy()

        self.train_acc_metric   = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric     = keras.metrics.SparseCategoricalAccuracy()

    def fit(self, train_dataset, validation_dataset, untrained_model, epochs = 5):
        for epoch in range(epochs):
            for step, (x_actual_train, y_actual_train_vector) in enumerate(train_dataset):
                with tf.GradientTape() as gradient_calc:
                    y_pred_prob_vector   = model(x_actual_train, training=True)
                    loss_value           = self.loss(y_actual_train_vector, y_pred_prob_vector)
                    # tf.keras.losses.categorical_crossentropy(y_actual_train_vector, y_pred_prob_vector, from_logits=False)
                    acc_value            = self.train_acc_metric(y_actual_train_vector, y_pred_probs)
                
                grads = gradient_calc.gradient(loss_value, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in validation_dataset:
                val_logits = model(x_batch_val, training=False)
                self.val_acc_metric.update_state(y_batch_val, val_logits)

trainer = Trainer()
trainer.fit(training_dataset, validation_dataset, model)