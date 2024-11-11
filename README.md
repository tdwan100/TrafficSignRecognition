# Traffic Sign Classification with CNNs

This project is a traffic sign classifier that uses Convolutional Neural Networks (CNNs) to classify images from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is designed to identify traffic signs in images, which can be useful for autonomous vehicles and other vision-based systems.

## Project Structure

The project consists of the following main components:

- **Data Preprocessing**: Functions to preprocess and prepare images for training.
- **Model Training**: Multiple training scripts implementing different CNN architectures.
- **Model Evaluation**: Scripts to evaluate the model’s accuracy on the test set.

## Files and Folders

| File                | Description                                                                                      |
|---------------------|--------------------------------------------------------------------------------------------------|
| `preprocess.py`     | Prepares the dataset by resizing, normalizing, and cropping images.                             |
| `settings.py`       | Stores global constants and hyperparameters like batch size, epochs, learning rate, etc.        |
| `train_baseline.py` | Trains a baseline CNN model without batch normalization.                                        |
| `train_enhanced1.py`| Trains a VGG16-based model, with an option to use pretrained weights.                           |
| `train_enhanced2.py`| Trains a baseline CNN model with batch normalization.                                           |
| `test_baseline.py`  | Tests the baseline model on the test set and computes accuracy.                                 |
| `test_loadmodel.py` | Loads a trained model from disk and tests it on the test dataset.                               |
| `model.py`          | Defines the CNN architectures: baseline, VGG16, and batch normalized models.                    |
| `GTSRB`             | Contains subfolders for training (`Final_Training`) and test data (`Final_Test`) from the GTSRB dataset. |

## Requirements

The following libraries are required:

- Python 3.6+
- Keras
- TensorFlow
- NumPy
- Scikit-Image
- Pandas

To install the dependencies, you can run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess the Data**

   Use the `preprocess.py` script to prepare the training data. Run:

   ```python
   from preprocess import preprocess
   X, Y = preprocess()
   ```

2. **Train the Model**

   Choose one of the training scripts to train a model.

   - **Baseline Model**: Run `train_baseline.py` to train a simple CNN.
     ```python
     from train_baseline import train_baseline
     model = train_baseline(X, Y)
     ```

   - **VGG16 Model**: Run `train_enhanced1.py` to train a VGG16-based model, with or without pretrained weights.
     ```python
     from train_enhanced1 import train_vgg16
     model = train_vgg16(X, Y, vgg16_pretrained=True)
     ```

   - **Batch Normalized Model**: Run `train_enhanced2.py` to train a CNN with batch normalization.
     ```python
     from train_enhanced2 import train_batch_normalized
     model = train_batch_normalized(X, Y)
     ```

   Each script will save the best model as an `.h5` file in the `trained_models` directory.

3. **Evaluate the Model**

   To test the model’s performance on the test set, use one of the following evaluation scripts:

   - **Baseline Model**: Run `test_baseline.py`.
   - **Load and Test a Saved Model**: Use `test_loadmodel.py` to load and test a saved model.

   For example:
   ```python
   from test_loadmodel import load_model, test_model
   model = load_model('./trained_models/baseline_30_32_0.01.h5')
   accuracy = test_model(model)
   print(f"Test accuracy = {accuracy:.2%}")
   ```

## Customizing Hyperparameters

You can modify hyperparameters such as batch size, learning rate, and epochs by editing `settings.py`. You can also use the `set_parameters` function to set them programmatically.

```python
from settings import set_parameters
set_parameters(batch_size=64, epochs=20, lr=0.001, decay=1e-5)
```

## CNN Architectures

The project includes three CNN architectures:

1. **Baseline Model**: A simple CNN without batch normalization (`train_baseline.py`).
2. **VGG16 Model**: A CNN based on VGG16, with an option for pretrained weights (`train_enhanced1.py`).
3. **Batch Normalized Model**: A CNN with batch normalization layers (`train_enhanced2.py`).

## Directory Structure

The directory structure for the GTSRB dataset should be organized as follows:

```plaintext
GTSRB/
├── Final_Training/
│   └── Images/
├── Final_Test/
│   └── Images/
└── GT-final_test.csv
```

## Example Output

During training, each model will display the loss and accuracy per epoch. The best-performing model on the validation set will be saved to the `trained_models` directory. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
