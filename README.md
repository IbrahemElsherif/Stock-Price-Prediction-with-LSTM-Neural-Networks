# Stock Price Prediction using LSTM

A deep learning project focused on predicting stock prices using LSTM (Long Short-Term Memory) neural networks. The model achieves a MAPE of 9.58% on test data, demonstrating good predictive capabilities for financial time series forecasting.

## 📊 Project Overview

This project implements a time series prediction model for stock prices using sequential data. The model uses historical price data to predict future stock prices, incorporating various technical indicators and proper data preprocessing techniques.

## 🔑 Key Features

- LSTM-based deep learning architecture
- Time series data preprocessing
- MinMax scaling for data normalization
- Sequential data preparation
- Performance metrics visualization
- Custom learning rate scheduler

## 📈 Model Architecture

```python
def Model():
    model = Sequential([
        LSTM(200, input_shape=(5, 1), activation=tf.nn.leaky_relu, return_sequences=True),
        LSTM(200, activation=tf.nn.leaky_relu),
        Dense(200, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(50, activation=tf.nn.leaky_relu),
        Dense(5, activation=tf.nn.leaky_relu)
    ])
    return model
```

## 📊 Performance Metrics

- RMSE: 13.96
- MAPE: 9.58%

## 🔧 Data Preprocessing

The project includes comprehensive data preprocessing:
- Sequence creation for time series data
- MinMax scaling
- Train-test split based on dates
- Custom dataset creation function

```python
def create_dataset(data, start_date, time_step=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = data['Adj. Close'][data['Date'] < start_date].to_numpy()
    test_data = data['Adj. Close'][data['Date'] >= start_date].to_numpy()
    # ... (sequence creation and scaling)
```

## 📉 Learning Rate Schedule

Custom learning rate scheduler implemented for optimal training:
- Warm-up phase
- Exponential decay
- Final fine-tuning phase

## 🚀 Getting Started

### Prerequisites
```
tensorflow>=2.0.0
numpy
pandas
scikit-learn
matplotlib
```

### Installation
```bash
git clone 
```
```colab
githubtocolab.com
```

### Usage
```python
# Load and preprocess data
train_x, train_y, test_x, test_y = create_dataset(data, '2020-10-12', time_step=5)

# Create and train model
model = Model()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
history = model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y))
```

## 📊 Results Visualization

The project includes visualization tools for:
- Training/Validation loss curves
- Actual vs Predicted price comparisons
- Performance metrics over time

## 🔄 Future Improvements

- Implement additional technical indicators
- Add attention mechanisms
- Explore ensemble methods
- Handle different price ranges more effectively
- Implement log transformation for price scaling


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/stock-price-prediction](https://github.com/yourusername/stock-price-prediction)


---
⭐️ If you found this project helpful, please give it a star!