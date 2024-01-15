import tensorflow as tf
import tensorflow.keras as keras # type:ignore

B,H,W,C      = 2, 224, 224, 3
single_batch = tf.random.normal((B,H,W,C))

NUM_CLASSES = 120
"""
    NN: Architecture
    Input Layer  = Shape (224, 224, 3)
    Hidden Layer = 240 Neurons
    Output Layer = 120 Neurons
"""
class simple_fcnn(keras.Model):
    def __init__(self):
        super().__init__()

        self.internal_model = tf.keras.models.Sequential([
            keras.layers.Input(shape = (224, 224, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(units = 240, activation="relu", ),
            keras.layers.Dense(units = NUM_CLASSES, activation="softmax"),
        ])

    def call(self, input_single_batch):
        final_layer_output = self.internal_model(input_single_batch)

        return final_layer_output


model  = simple_fcnn()
output = model(single_batch)
print(output)

assert len(output.shape) == len("BD")
assert output.shape[0]   == B
assert output.shape[1]   == NUM_CLASSES