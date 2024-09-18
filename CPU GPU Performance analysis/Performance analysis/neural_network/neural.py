import numpy as np
import logging
import psutil
import GPUtil
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json

# Define the path for the text file
log_file = 'C:/Users/21meh/OneDrive/Desktop/stats_visua/system-vitals-dashboard/backend/static/training_logs.txt'

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

class NeuralNetwork:
    def __init__(self, units_per_layer_list, epochs, iteration, learn_rate, validation_data=None):
        self.weights = []
        self.bias = []
        self.z = []  # weights * inputs + bias for units in each layer
        self.a = []  # activation function applied to z for units in each layer
        self.alpha = learn_rate
        self.iteration = iteration
        self.epochs = epochs
        self.validation_data = validation_data

        l = len(units_per_layer_list)
        for i in range(l - 1):
            self.weights.append(np.random.randn(units_per_layer_list[i + 1], units_per_layer_list[i]))
            self.bias.append(np.random.randn(units_per_layer_list[i + 1]))

    def dtanh(self, x):
        return 1.0 - np.tanh(x) ** 2

    def forward(self, input, label):
        self.a = [input]
        self.z = []
        
        for w, b in zip(self.weights, self.bias):
            z = np.dot(self.a[-1], w.T) + b
            self.z.append(z)
            input = np.tanh(z)
            self.a.append(input)
        
        output = self.a[-1]
        self.error = 0.5 * np.power(output - label, 2)
        self.derror = output - label
        return output

    def backpropagate(self):
        delta = self.derror * self.dtanh(self.z[-1])
        self.weights[-1] -= self.alpha * np.outer(delta, self.a[-2])
        self.bias[-1] -= self.alpha * delta

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(delta, self.weights[i + 1]) * self.dtanh(self.z[i])
            self.weights[i] -= self.alpha * np.outer(delta, self.a[i])
            self.bias[i] -= self.alpha * delta

    def calculate_metrics(self, y_true, y_pred):
        try:
            # Convert continuous predictions to binary classes if necessary
            y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
            y_true_binary = (np.array(y_true) > 0.5).astype(int)
            
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=1)
            recall = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=1)
            f1 = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=1)
            
            return accuracy, precision, recall, f1
        except Exception:
            return None, None, None, None

    def get_system_metrics(self):
        try:
            gpu_metrics = {
                'gpu_usage': self.get_gpu_usage(),
                'gpu_temperature': self.get_gpu_temperature()
            }
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=0.1)  # Adjust interval if needed
            disk_io = psutil.disk_io_counters()
            
            return {
                'gpu_usage': gpu_metrics['gpu_usage'],
                'gpu_temperature': gpu_metrics['gpu_temperature'],
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'disk_io': disk_io.read_bytes + disk_io.write_bytes
            }
        except Exception:
            return {
                'gpu_usage': None,
                'gpu_temperature': None,
                'memory_usage': None,
                'cpu_usage': None,
                'disk_io': None
            }

    def get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # Return GPU usage as percentage
        except Exception:
            return None
        return None

    def get_gpu_temperature(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].temperature  # Return GPU temperature
        except Exception:
            return None
        return None

    def has_gpu(self):
        try:
            return len(GPUtil.getGPUs()) > 0
        except Exception:
            return False

    def train(self, input_data, input_labels):
        for epoch in range(self.epochs):  # Change to epoch loop
            for i in range(self.iteration):
                average_error = 0
                for d, l in zip(input_data, input_labels):
                    self.forward(d, l)
                    average_error += self.error.item()  # Convert to scalar
                    self.backpropagate()
                    self.a = []
                    self.z = []

                avg_error = average_error / len(input_data)

                # Validation metrics
                val_accuracy, val_precision, val_recall, val_f1 = None, None, None, None
                if self.validation_data:
                    val_data, val_labels = self.validation_data
                    val_predictions = [self.forward(d, 0) for d in val_data]  # Forward pass for validation data
                    val_accuracy, val_precision, val_recall, val_f1 = self.calculate_metrics(val_labels, val_predictions)

                system_metrics = self.get_system_metrics()

                # Log essential data in a JSON-friendly format
                log_entry = {
                    "epoch": epoch + 1,
                    "iteration": i + 1,
                    "average_error": avg_error,
                    "validation_accuracy": val_accuracy,
                    "precision": val_precision,
                    "recall": val_recall,
                    "f1_score": val_f1,
                    "system_metrics": system_metrics
                }

                try:
                    logging.info(json.dumps(log_entry))
                except Exception:
                    pass  # Suppress errors during logging to avoid affecting the training process

    def predict(self, input_data):
        output = self.forward(input_data, 0)
        print(output)
        self.error = 0
        self.a = []
        self.z = []

# Example usage with more data and iterations
if __name__ == "__main__":
    input_data = [
        [0, 0], [0, 1], [1, 0], [1, 1],  # Example for XOR problem
        [0.5, 0.5], [0.5, 1], [1, 0.5], [1, 1]  # Additional data points
    ]
    input_labels = [
        [0], [1], [1], [0],  # Corresponding XOR outputs
        [0.5], [1], [1], [0.5]  # Corresponding labels for additional data points
    ]

    # Example validation data
    validation_data = (input_data, input_labels)  # For simplicity, using the same data

    # Instantiate and train the neural network
    m = NeuralNetwork([2, 4, 2, 1], epochs=10, iteration=1000, learn_rate=0.1, validation_data=validation_data)
    m.train(input_data, input_labels)
