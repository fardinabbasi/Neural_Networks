# Neural Networks
> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/info.svg">
>   <img alt="Info" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/info.svg">
> </picture><br>
>
> Check the branches!
# [Multi Layer Perceptron](https://github.com/fardinabbasi/Neural_Networks/tree/MLP)
**ECG signals** are classified into four classes: N for Normal, A for Atrial fibrillation, O for Others, and ~ for cases affected by unknown noise. 
The dataset used for classification is "[ECG.csv](https://github.com/fardinabbasi/Neural_Networks/blob/MLP/ECG.csv)" which contains 169 features.
### Data Exploration
The class distribution of the samples is as follows: N (Normal) - 5992 samples, O (Others) - 3151 samples, A (Atrial fibrillation) - 923 samples, and ~ (Affected with unknown noise) - 187 samples.

<img src="/readme_images/mlp_exploration.png">

Indeed, the class distribution appears to be **highly imbalanced**.
### Preprocessing
The dataset has been divided into a **training set** and a **test set**. 
Additionally, the features have been normalized using **StandardScaler**.
### MLP Architecture
Here is the architecture of the MLP (Multi-Layer Perceptron) classifier:
```ruby
clf = MLPClassifier(hidden_layer_sizes=(128,64,32,16), activation='relu', solver='sgd', max_iter=1000, random_state=42)
```
### Comparing Performance with Normalized and Unnormalized Features
The model was fed with two different types of features: **normalized** features and **unnormalized** features. Below are the results for comparison:
| Features | Classification Report | Confusion Matrix |
| --- | --- | --- |
| Unnormalized | <img src="/readme_images/mlp_wo_r.jpg"> | <img src="/readme_images/mlp_wo_c.png"> |
| Normalized | <img src="/readme_images/mlp_w_r.jpg"> | <img src="/readme_images/mlp_w_c.png"> |

It is clear from the results that normalizing has quite an impressive impact on model performance. Normalization is a crucial preprocessing step because a feature with a higher range can **outweigh** a feature with a lower range. [Read More](https://sathish-manthani.medium.com/data-normalization-and-standardization-7ce8cb6472ae)
## Data Manipulation
To handle the imbalanced distribution of classes, the class ~, which has the lowest number of samples, is **omitted**, and the A and O classes are **merged** into an '**abnormal**' class.
| Classification Report | Confusion Matrix |
| --- | --- |
| <img src="/readme_images/mlp_m_r.jpg"> | <img src="/readme_images/mlp_m_c.png"> |

The results indicate a slight improvement in model performance.
# [Convolutional Neural Networks](https://github.com/fardinabbasi/Neural_Networks/tree/CNN)
Performing a CNN model on the [CIFAR-10](https://keras.io/api/datasets/cifar10/) image dataset, which consists of 10 classes and a total of 60,000 images.
### Data Exploration
Here are 5 random images from the CIFAR-10 image dataset.
| Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | --- | --- | --- | --- |
| <img src="/readme_images/s1.png"> | <img src="/readme_images/s2.png"> | <img src="/readme_images/s3.png"> | <img src="/readme_images/s4.png"> | <img src="/readme_images/s5.png"> |
### Preprocessing
The dataset has been split into a **training set** and a **test set**. 
Furthermore, the labels have been one-hot encoded using the following code snippet.
```ruby
from tensorflow.keras.utils import to_categorical
```
```ruby
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.8, random_state=42)
```
### CNN Architecture
Here is the architecture of the CNN (Convolutional Neural Networks) classifier:
```ruby
from tensorflow import keras
```
```ruby
cnn = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu",  input_shape=x_train.shape[1:]),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

cnn.summary()
```
### Comparing Performance with Different Optimizers
Here are the model results obtained using three different optimizers, namely [SGD](https://keras.io/api/optimizers/sgd/), [Adam](https://keras.io/api/optimizers/adam/), and [RMSprop](https://keras.io/api/optimizers/rmsprop/).
| Optimizer | Training Loss & Precision | Validation Loss & Precision | Classification Report |
| --- | --- | --- | --- |
| SGD | <img src="/readme_images/sgd_t.png"> | <img src="/readme_images/sgd_v.png"> | <img src="/readme_images/sgd_r.jpg"> |
| Adam | <img src="/readme_images/adam_t.png"> | <img src="/readme_images/adam_v.png"> | <img src="/readme_images/adam_r.jpg"> |
| RMSprop | <img src="/readme_images/rsm_t.png"> | <img src="/readme_images/rms_v.png"> | <img src="/readme_images/rms_r.jpg"> |

The above results demonstrate the importance of choosing an optimizer wisely.
Now, let's provide a brief explanation of the application of each optimizer:

1. **SGD** is generally suitable for **shallow networks** or when **memory resources** are limited. However, it may struggle with **saddle points** and **flat regions** in the loss landscape.
2. **Adam** is widely used and **generally recommended** as a good default optimizer for most neural network architectures. It often provides **faster convergence** and **better performance** than SGD.
3.  **RMSprop** is **less sensitive** to the **learning rate** hyperparameter compared to SGD, making it more suitable for various tasks and network architectures.
