
#%%
import tensorflow as tf
from tensorflow import keras

dataset_path = "/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/kaggle_dog_breed/src/dataset/dog-breed-imagefolder"
training_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(dataset_path, 
                                            image_size = (224, 224),  # Default value is (256, 256). **source of potential error**
                                            validation_split = 0.1, subset = "both", seed = 10, # Need to set subset & seed both for validation_split
                                            batch_size = 1, shuffle= True, # Default values used automatically. **SOURCE OF POTENTIAL ERROR**
                                            color_mode = "rgb") # Channels = 3. Hidden from us

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
class Trainer():
    def __init__(self):
        self.optimizer          = keras.optimizers.Adam()
        self.loss               = keras.losses.CategoricalCrossentropy()

        self.train_acc_metric   = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric     = keras.metrics.SparseCategoricalAccuracy()

    def fit(self, train_dataset, validation_dataset, untrained_model, epochs = 5):
        for epoch in range(epochs):
            for step, (x_actual_train, y_actual_train) in enumerate(train_dataset):
                with tf.GradientTape() as gradient_calc:
                    y_actual_one_hot_vec = tf.one_hot(y_actual_train, depth= 120)
                    y_pred_prob_vector   = model(x_actual_train, training=True)

                    loss_value           = self.loss(y_actual_one_hot_vec, y_pred_prob_vector)
                    # tf.keras.losses.categorical_crossentropy(y_actual_one_hot_vec, y_pred_prob_vector, from_logits=False)
                    acc_value       = self.train_acc_metric(y_actual_train, y_pred_probs)
                
                grads = gradient_calc.gradient(loss_value, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in validation_dataset:
                val_logits = model(x_batch_val, training=False)
                self.val_acc_metric.update_state(y_batch_val, val_logits)

trainer = Trainer()
trainer.fit(training_dataset, validation_dataset, model)