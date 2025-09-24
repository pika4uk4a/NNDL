"""
Нейросетевой фреймворк для машинного обучения

Реализован собственный нейросетевой фреймворк для обучения полносвязных нейронных сетей
с расширенными возможностями для решения задач классификации и регрессии.

Основные возможности:
- Создание многослойных нейросетей
- Обработка данных (map, minibatching, перемешивание)
- Алгоритмы оптимизации (SGD, Momentum SGD, RMSProp)
- Функции активации (ReLU, Sigmoid, Tanh, Softmax)
- Функции потерь (CrossEntropyLoss, MSE, MAE)
- Простое обучение "в несколько строк"
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification, load_iris, make_regression
import numpy as np

np.random.seed(0)
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split


class BaseOptimizer:
    """Базовый класс для всех оптимизаторов"""
    def optimize_parameters(self, learning_rate, weights, bias, dweights, dbias):
        """Обновляет параметры модели на основе градиентов"""
        pass


class StochasticGradientDescent(BaseOptimizer):
    """Стохастический градиентный спуск (SGD)"""
    def optimize_parameters(self, learning_rate, weights, bias, dweights, dbias):
        weights -= learning_rate * dweights
        bias -= learning_rate * dbias
        return weights, bias


class MomentumOptimizer(BaseOptimizer):
    """Оптимизатор с моментом (Momentum SGD)"""
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity_weights = None
        self.velocity_bias = None

    def optimize_parameters(self, learning_rate, weights, bias, dweights, dbias):
        if self.velocity_weights is None:
            self.velocity_weights = np.zeros_like(dweights)
        if self.velocity_bias is None:
            self.velocity_bias = np.zeros_like(dbias)

        self.velocity_weights = self.momentum * self.velocity_weights + learning_rate * dweights
        weights -= self.velocity_weights

        self.velocity_bias = self.momentum * self.velocity_bias + learning_rate * dbias
        bias -= self.velocity_bias

        return weights, bias


class RMSPropOptimizer(BaseOptimizer):
    """RMSProp оптимизатор"""
    def __init__(self, rho=0.9, epsilon=1e-8):
        self.rho = rho
        self.epsilon = epsilon
        self.cache_weights = None
        self.cache_bias = None

    def optimize_parameters(self, learning_rate, weights, bias, dweights, dbias):
        if self.cache_weights is None:
            self.cache_weights = np.zeros_like(dweights)
        if self.cache_bias is None:
            self.cache_bias = np.zeros_like(dbias)

        self.cache_weights = self.rho * self.cache_weights + (1 - self.rho) * np.square(dweights)
        weights -= learning_rate * dweights / (np.sqrt(self.cache_weights) + self.epsilon)

        self.cache_bias = self.rho * self.cache_bias + (1 - self.rho) * np.square(dbias)
        bias -= learning_rate * dbias / (np.sqrt(self.cache_bias) + self.epsilon)

        return weights, bias


# Функции потерь для классификации
class CrossEntropyLoss:
    """Функция потерь перекрестной энтропии для многоклассовой классификации"""
    def compute_loss(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        p_of_y = predictions[np.arange(len(targets)), targets]
        log_prob = - np.log(p_of_y)
        return log_prob.mean()

    def compute_gradients(self, loss):
        dlog_softmax = np.zeros_like(self.predictions)
        dlog_softmax[np.arange(len(self.targets)), self.targets] -= 1.0 / len(self.targets)
        return dlog_softmax / self.predictions


class BinaryCrossEntropyLoss:
    """Функция потерь бинарной перекрестной энтропии"""
    def compute_loss(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        p_of_y = predictions[np.arange(len(targets)), targets]
        return (-(targets * np.log(p_of_y) + (1 - targets) * np.log(1 - p_of_y))).mean()

    def compute_gradients(self, loss):
        dbin = np.zeros_like(self.predictions)
        dbin[np.arange(len(self.targets)), self.targets] -= 1.0 / len(self.targets)
        return (((1 - dbin) / (1 - self.predictions)) + (dbin / self.predictions))


# Функции потерь для регрессии
class MeanSquaredErrorLoss:
    """Функция потерь среднеквадратичной ошибки"""
    def compute_loss(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean(np.power(self.predictions.flatten() - targets, 2))

    def compute_gradients(self, loss):
        self.targets = self.targets.reshape(-1, 1)
        return 2 * (self.predictions - self.targets)


class MeanAbsoluteErrorLoss:
    """Функция потерь средней абсолютной ошибки"""
    def compute_loss(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean(np.abs(self.predictions.flatten() - targets))

    def compute_gradients(self, loss):
        self.targets = self.targets.reshape(-1, 1)
        return np.sign(self.predictions - self.targets)


class NeuralNetwork:
    """Основной класс нейронной сети"""
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, input_data):
        """Прямой проход через сеть"""
        for layer in self.layers:
            input_data = layer.forward_pass(input_data)
        return input_data

    def backward_pass(self, gradient):
        """Обратный проход через сеть"""
        for layer in self.layers[::-1]:
            gradient = layer.backward_pass(gradient)
        return gradient

    def update_parameters(self, learning_rate, optimizer):
        """Обновление параметров сети"""
        for layer in self.layers:
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(learning_rate, optimizer)


def apply_function_to_data(func, data):
    """Применяет функцию к каждому элементу данных"""
    for item in data:
        yield func(item)


def shuffle_dataset(features, labels, num_samples):
    """Перемешивает датасет"""
    combined_data = np.c_[features.reshape(num_samples, -1), labels.reshape(num_samples, 1)]
    rng = np.random.default_rng(seed=None)
    rng.shuffle(combined_data)
    return combined_data


class DeepLearningFramework:
    """Основной класс фреймворка для глубокого обучения"""

    def build_network(self, layers, loss_function, optimizer=StochasticGradientDescent(),
                     learning_rate=0.02, epochs_count=20,
                     classification=True, regression=False):
        """Строит нейронную сеть с заданными параметрами"""
        self.network = NeuralNetwork(layers)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs_count = epochs_count
        self.is_classification = classification
        self.is_regression = regression

        if self.learning_rate == 0.02:
            if not isinstance(self.optimizer, StochasticGradientDescent):
                self.learning_rate = 0.001

        if self.is_regression == True:
            self.is_classification = False

    def _forward_propagation(self, input_batch, target_batch, network, loss_function):
        """Выполняет прямой проход и вычисляет потери"""
        predictions = network.forward_pass(input_batch)
        loss = loss_function.compute_loss(predictions, target_batch.flatten())
        return predictions, loss

    def _backward_propagation(self, loss, learning_rate, network, loss_function, optimizer):
        """Выполняет обратный проход и обновляет параметры"""
        gradient = loss_function.compute_gradients(loss)
        network.backward_pass(gradient)
        network.update_parameters(learning_rate, optimizer)

    def _calculate_accuracy(self, network, input_data, target_data):
        """Вычисляет точность для задач классификации"""
        predictions = network.forward_pass(input_data)
        predicted_classes = np.argmax(predictions, axis=1)
        return (predicted_classes == target_data.flatten()).mean()

    def train_model(self, training_features, training_labels, batch_size=8):
        """Обучает модель на предоставленных данных"""
        features, labels = np.array(training_features), np.array(training_labels)
        num_samples = features.shape[0]

        batch_count = int(num_samples // batch_size)

        print('Обучение модели:')
        for epoch in range(self.epochs_count):
            shuffled_data = shuffle_dataset(features, labels, num_samples)

            for batch in range(batch_count):
                start = batch * batch_size
                stop = start + batch_size

                batch_features = shuffled_data[start:stop, :-1]
                batch_labels = shuffled_data[start:stop, -1:]

                if self.is_classification == True:
                    batch_labels = batch_labels.astype('int32')

                predictions, loss = self._forward_propagation(batch_features, batch_labels, 
                                                           self.network, self.loss_function)
                self._backward_propagation(loss, self.learning_rate, self.network, 
                                        self.loss_function, self.optimizer)

            if self.is_classification == True:
                accuracy = self._calculate_accuracy(self.network, shuffled_data[:, :-1], shuffled_data[:, -1:])
                print(f"Эпоха {epoch + 1}: точность = {accuracy}")
            elif self.is_regression == True:
                print(f"Эпоха {epoch + 1}: потери = {loss}")

    def make_predictions(self, test_features, test_labels):
        """Делает предсказания на тестовых данных"""
        num_samples = test_features.shape[0]
        combined_data = np.c_[test_features.reshape(num_samples, -1), test_labels.reshape(num_samples, 1)]
        test_features = combined_data[:, :-1]
        test_labels = combined_data[:, -1:]

        if self.is_classification == True:
            test_labels = test_labels.astype('int32')

        predictions = []

        for i in range(num_samples):
            pred, loss = self._forward_propagation(test_features, test_labels, 
                                                self.network, self.loss_function)

        if self.is_classification == True:
            predicted_classes = np.argmax(pred, axis=1)
            accuracy = (predicted_classes == test_labels.astype('int32').flatten()).mean()
            return predicted_classes, accuracy

        elif self.is_regression == True:
            predictions.append(pred.flatten().tolist())
            return np.array(predictions).flatten(), ((np.array(predictions).flatten() - test_labels.flatten()) ** 2).mean()


class DenseLayer:
    """Полносвязный слой нейронной сети"""
    def __init__(self, input_size, output_size):
        self.weights = np.random.normal(0, 1.0 / np.sqrt(input_size), (output_size, input_size))
        self.bias = np.zeros((1, output_size))

    def forward_pass(self, input_data):
        """Прямой проход через слой"""
        self.input_data = input_data
        return np.dot(input_data, self.weights.T) + self.bias

    def backward_pass(self, gradient):
        """Обратный проход через слой"""
        self.dweights = np.dot(gradient.T, self.input_data)
        self.dbias = gradient.sum(axis=0)
        return np.dot(gradient, self.weights)

    def update_parameters(self, learning_rate, optimizer):
        """Обновляет параметры слоя"""
        self.weights, self.bias = optimizer.optimize_parameters(learning_rate, self.weights, 
                                                              self.bias, self.dweights, self.dbias)


class SoftmaxActivation:
    """Функция активации Softmax"""
    def forward_pass(self, input_data):
        self.input_data = input_data
        input_max = input_data.max(axis=1, keepdims=True)
        exp_input = np.exp(input_data - input_max)
        sum_exp = exp_input.sum(axis=1, keepdims=True)
        return exp_input / sum_exp

    def backward_pass(self, gradient):
        predictions = self.forward_pass(self.input_data)
        predictions_gradient = predictions * gradient
        return predictions_gradient - predictions * predictions_gradient.sum(axis=1, keepdims=True)


class SigmoidActivation:
    """Функция активации Sigmoid"""
    def forward_pass(self, input_data):
        self.output = 1.0 / (1.0 + np.exp(-input_data))
        return self.output

    def backward_pass(self, gradient):
        return gradient * self.output * (1.0 - self.output)


class ThresholdActivation:
    """Функция активации Threshold"""
    def forward_pass(self, input_data, threshold=0):
        self.output = np.zeros_like(input_data)
        self.output[input_data > threshold] = 1.0
        return self.output

    def backward_pass(self, gradient):
        return gradient


class TanhActivation:
    """Функция активации Tanh"""
    def forward_pass(self, input_data):
        output = np.tanh(input_data)
        self.output = output
        return output

    def backward_pass(self, gradient):
        return (1.0 - self.output ** 2) * gradient


class ReLUActivation:
    """Функция активации ReLU"""
    def forward_pass(self, input_data):
        self.output = np.maximum(0, input_data)
        return self.output

    def backward_pass(self, gradient):
        return np.multiply(gradient, np.int64(self.output > 0))


class DataLoader:
    """Класс для загрузки и обработки датасетов"""
    def __init__(self, dataset_name, test_size=0.2, random_state=42):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state

        # Загрузка и обработка данных
        self.training_features, self.test_features, self.training_labels, self.test_labels = self._load_dataset()

    def _load_dataset(self):
        """Загружает указанный датасет"""
        if self.dataset_name == "MNIST":
            # Загрузка MNIST
            (training_features, training_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()
            training_features = training_features.reshape(training_features.shape[0], 28, 28, 1)
            test_features = test_features.reshape(test_features.shape[0], 28, 28, 1)
            # Нормализация изображений в диапазон [0, 1]
            training_features = training_features.astype("float32") / 255
            test_features = test_features.astype("float32") / 255

        elif self.dataset_name == "FashionMNIST":
            # Загрузка FashionMNIST
            (training_features, training_labels), (test_features, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            training_features = training_features.reshape(training_features.shape[0], 28, 28, 1)
            test_features = test_features.reshape(test_features.shape[0], 28, 28, 1)
            # Нормализация изображений в диапазон [0, 1]
            training_features = training_features.astype("float32") / 255
            test_features = test_features.astype("float32") / 255
        elif self.dataset_name == "Iris":
            # Загрузка Iris
            iris = datasets.load_iris()
            features, labels = iris.data[:, :3], iris.target
            training_features, test_features, training_labels, test_labels = train_test_split(features, labels,
                                                                test_size=0.2,
                                                                random_state=0)
        else:
            raise ValueError("Неподдерживаемое имя датасета. Выберите 'MNIST', 'FashionMNIST' или 'Iris'.")

        return training_features, test_features, training_labels, test_labels

    def get_training_data(self, num_samples=None):
        """Возвращает обучающие данные"""
        if num_samples is not None:
            return self.training_features[:num_samples], self.training_labels[:num_samples]
        return self.training_features, self.training_labels

    def get_test_data(self, num_samples=None):
        """Возвращает тестовые данные"""
        if num_samples is not None:
            return self.test_features[:num_samples], self.test_labels[:num_samples]
        return self.test_features, self.test_labels



