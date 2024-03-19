# Notes of chapter 2, "Deep Learning"

Structured vs Unstructured data
Deep learning can be applied to both structured and unstructured data.

`Deep learning` has now almost become synonymous with `deep neural networks`

## MLPs(Multilayer perceptron)
- Neural networks where all adjacent layers are fully connected.

## Sequential model and functional API
- Sequential model is used for simple models
- Prefer functional API

## Activation functions
- It's used to introduce non-linearity to the model.
- ReLU = max(0, x)
- Leaky ReLU = max(0.1x, x)
- Sigmoid = 1 / (1 + exp(-x))
- Softmax = exp(x) / sum(exp(x))

## Inspecting the model
```python
from tensorflow.keras import layers, models
input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation='relu')(x)
x = layers.Dense(units=150, activation='relu')(x)
output_layer = layers.Dense(units=10, activation='softmax')(x)
model = models.Model(input_layer, output_layer)

model.summary()
```

## Understanding the model
### Understanding how the number of parameters are calculated
Each unit within a given layer is also connected to one additional bias unit that always output 1.
This ensures that the output of the layer is not always zero when the input is zero.

## Compiling the model
Compile the model with:
1. An optimizer
2. A loss function

### Most commonly used loss functions
1. Mean squared error (MSE)
   * (1/n) * sum((y_true - y_pred)^2)
2. Binary cross-entropy
   * Binary classification problem
   * Multi-label problem
3. Categorical cross-entropy
   * -sum(y_true * log(y_pred))
   * Sum on the number of classes
   * The goal is to maximize the probability of the correct class.

Understanding the loss function:
1. MSE: Used for regression problem, where the output is a continuous value.
2. Binary cross-entropy: Used for binary classification problems.
3. Categorical cross-entropy: Used for classification problems where each observation belongs to one class.