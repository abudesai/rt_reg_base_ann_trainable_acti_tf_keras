
import numpy as np, pandas as pd
import joblib
import sys
import math
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback



model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"

MODEL_NAME = "reg_base_ann_with_trainable_activation"

COST_THRESHOLD = float('inf')


class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("Cost is inf, so stopping training!!")
            self.model.stop_training = True


def get_init_values(shape): 
    dim = np.prod(shape)
    vals = np.random.randn(dim) / np.sqrt(dim)
    return vals.reshape(shape)


class EarlyStoppingAtMinLoss(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self, monitor, patience=3, min_epochs=50):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = min_epochs
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch >= self.min_epochs:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # print("Restoring model weights from the end of the best epoch.")
                # self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            
            
class TrainableActivationLayer(Layer):
    def __init__(self, num_modes):
        super(TrainableActivationLayer, self).__init__()
        self.num_modes = num_modes        
            
    def build(self, input_shape): 
        D = input_shape[-1]
        shape = (1,D,self.num_modes)
        self.slopes = tf.Variable( initial_value = get_init_values(shape), dtype=tf.float32 )  # shape => (1, D, num_modes)
        self.intercepts = tf.Variable( initial_value = get_init_values(shape), dtype=tf.float32 )  # shape => (1, D, num_modes)
        self.locations = tf.Variable( initial_value = get_init_values(shape), dtype=tf.float32 )  # shape => (1, D, num_modes)
        shape = (1,D,1)
        self.lambda_ = tf.Variable( initial_value = get_init_values(shape), dtype=tf.float32 )    # shape => (1, D, 1)        
    
    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=2, name="expand_dims")  # shape goes from (NxD) => (N, D, 1)        
        sq_diff = tf.math.square(inputs - self.locations, name="sq_diff") # shape = (N, D, num_modes)
        exp = tf.math.exp( - self.lambda_ * sq_diff, name="exp")      # shape = (N, D, num_modes)
        probs = tf.nn.softmax(exp, name="softmax")      # shape = (N, D, num_modes)
        
        y_multi = tf.math.multiply(inputs , self.slopes) + self.intercepts
        y_multi_x_probs = y_multi * probs
        
        output_ = tf.math.reduce_sum(y_multi_x_probs, axis=-1, name="output")         # shape = (N, D)
        return output_


class Regressor():     
    def __init__(self, D, l1_reg=1e-3, l2_reg=1e-1, lr = 1e-3, num_cps=3, **kwargs) -> None:
        self.D = D
        self.l1_reg = np.float(l1_reg)
        self.l2_reg = np.float(l2_reg)
        self.lr = lr
        self.num_cps = num_cps
        
        self.model = self.build_model()
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.lr),
            # optimizer=SGD(learning_rate=self.lr),
            metrics=['mse'],
        )
        
        
    def build_model(self): 
        input_ = Input(self.D)
        reg = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        
        x = input_        
        
        x = Dense(max(3, self.D//3), activity_regularizer=reg)(x)
        x = TrainableActivationLayer(num_modes = self.num_cps)(x)
        
        output_ = Dense(1, activity_regularizer=reg )(x)

        model = Model(input_, output_)
        # model.summary() ; sys.exit()
        return model
    
    
    def fit(self, train_X, train_y, valid_X=None, valid_y=None,
            batch_size=256, epochs=100, verbose=0):        
        
        if valid_X is not None and valid_y is not None:
            early_stop_loss = 'val_loss' 
            validation_data = [valid_X, valid_y]
        else: 
            early_stop_loss = 'loss'
            validation_data = None   
        
        early_stop_callback = EarlyStoppingAtMinLoss(monitor=early_stop_loss, patience=5, min_epochs=50)    
        
        infcost_stop_callback = InfCostStopCallback()
    
        history = self.model.fit(
                x = train_X,
                y = train_y, 
                batch_size = batch_size,
                validation_data=validation_data,
                epochs=epochs,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )
        return history
    
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X, verbose=verbose)
        return preds 
    

    def summary(self):
        self.model.summary()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.evaluate(x_test, y_test, verbose=0)        

    
    def save(self, model_path): 
        model_params = {
            "D": self.D,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "num_cps": self.num_cps,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        self.model.save_weights(os.path.join(model_path, model_wts_fname))


    @classmethod
    def load(cls, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        ann_model = cls(**model_params)
        ann_model.model.load_weights(os.path.join(model_path, model_wts_fname)).expect_partial()
        return ann_model


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = Regressor.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)


def get_data_based_model_params(data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    return {"D": data.shape[1]}