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

The dataset has been divided into a **training set** and a **test set**.

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
### Data Exploration
| Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | --- | --- | --- | --- |
| <img src="/readme_images/s1.png"> | <img src="/readme_images/s2.png"> | <img src="/readme_images/s3.png"> | <img src="/readme_images/s4.png"> | <img src="/readme_images/s5.png"> |

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
| Optimizer | Training Loss & Precision | Validation Loss & Precision | Classification Report |
| --- | --- | --- | --- |
| SGD | <img src="/readme_images/sgd_t.png"> | <img src="/readme_images/sgd_v.png"> | <img src="/readme_images/sgd_r.jpg"> |
| Adam | <img src="/readme_images/adam_t.png"> | <img src="/readme_images/adam_v.png"> | <img src="/readme_images/adam_r.jpg"> |
| RMSprop | <img src="/readme_images/rsm_t.png"> | <img src="/readme_images/rms_v.png"> | <img src="/readme_images/rms_r.jpg"> |
