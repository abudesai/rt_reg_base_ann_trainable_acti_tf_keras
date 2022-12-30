ANN regression with Trainable Activation in TensorFlow/Keras for Regression.

- elastic net
- tensorflow
- keras
- regularization
- activity regularizer
- activation layer
- neural network
- adam optimizer
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker

This is a neural network regressor with a single hidden layer. Model uses L1 and L2 regularization. The hidden middle layer uses a custom trainable activation layer that adjusts the shape of non-linearity of the activation function.

Adam optimizer is used to reduce the overall loss and improve accuracy of the model.

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT includes choosing the optimal values for learning rate for the Adam optimizer, L1 and L2 regularization and the number of change points for the trainable activation layer.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include Tensorflow and Keras for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.
