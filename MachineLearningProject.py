import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Load the dataset
def load_dataset(path):

    try:
        notify_loading_process()  # inform the user that the data started the loading process.
        data_chunks = process_chunks(path)  # Process the dataset in chunks.
        complete_dataset = combine_chunks(data_chunks)  # Combine all chunks.
        display_success_message(complete_dataset)  # Display success information.
        return complete_dataset
    except Exception as error:
        handle_loading_error(error)  # Handle any errors that occur.
        return None

# Helper functions

def notify_loading_process():

    print("Loading dataset in chunks (smaller parts) ...")

def process_chunks(file_path):

    # Initialize an empty list called chunck_List to store each chunk after being processed.
    chunk_list = []

    try:
        # Create a chunk iterator variable which reads the dataset in chunks.
        chunk_iterator = pd.read_csv(file_path, header=None, engine='python', chunksize=5000)

        # for each chunk do the following
        for index, chunk in enumerate(chunk_iterator):
            # prompt that the chunk is being processed.
            print(f"Processing chunk {index + 1}")
            # add the current chunk to the chunk list.
            chunk_list.append(chunk)
    except Exception as e:
        # if any abnormal thing occured ofr problem faced, print an error message to let the user know.
        print(f"Oops, unfortunately there is an error while processing chunks: {e}")
        # raise the problem to be handled later.
        raise

    # The complete list of chunks are returned after being processed chunk by chunk.
    return chunk_list


def combine_chunks(chunks):

    try:
        # Use pandas' concat method to combine the list of DataFrame chunks into one DataFrame.
        # The ignore_index=True parameter ensures that the resulting DataFrame has a continuous index.
        combined_dataset = pd.concat(chunks, ignore_index=True)
        # Return the combined DataFrame after successful concatenation.
        return combined_dataset
    except Exception as e:
        # Print an error message if an exception occurs during the concatenation process.
        print(f"Error while combining chunks: {e}")
        # Re-raise the exception to inform the caller of the issue.
        raise

def display_success_message(dataset):

    # Print a success message indicating that the dataset was loaded successfully.
    print("Dataset Loaded Successfully")
    # Print the shape of the dataset (rows and columns) for user information.
    print("Dataset Shape:", dataset.shape)

def handle_loading_error(error):

    # Print an error message describing what went wrong during the loading process.
    print(f"Error loading dataset: {error}")

####################################################

# Explore the dataset
def explore_dataset(dataset):

    if dataset is None:
        print("Dataset not loaded. Cannot explore.")
        return

    labels = dataset.iloc[:, 0]  # Assuming the first column contains the labels
    print("Unique Values in Label Column:", labels.unique())

    # Validate if labels are correct
    unique_classes = labels.unique()
    class_distribution = labels.value_counts()
    print("Number of Unique Classes:", len(unique_classes))
    print("Class Distribution:")
    print(class_distribution)

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_distribution.index, class_distribution.values, color='skyblue')
    plt.xlabel("Class (Alphabet)")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.xticks(unique_classes, [chr(int(c) + 65) for c in unique_classes])
    plt.show()

# Normalize the images
def normalize_images(dataset):

    if dataset is None:
        print("Dataset not loaded. Cannot normalize.")
        return None, None

    features = dataset.iloc[:, 1:]  # All columns except the first (labels)
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    print("Images Normalized")
    return pd.DataFrame(normalized_features), dataset.iloc[:, 0]

# Display sample images
def display_sample_images(normalized_features, labels):

    if normalized_features is None or labels is None:
        print("No data available to display images.")
        return

    num_samples = 10  # Number of samples to display
    sample_indices = np.random.choice(normalized_features.index, num_samples, replace=False)
    plt.figure(figsize=(12, 6))
    for idx, sample_idx in enumerate(sample_indices):
        image_array = normalized_features.iloc[sample_idx].to_numpy().reshape(28, 28)
        plt.subplot(2, 5, idx + 1)
        plt.imshow(image_array, cmap='gray')
        plt.title(f"Label: {chr(int(labels.iloc[sample_idx]) + 65)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def select_random_subset(data_features, data_labels, subset_count=10000):

    random_indices = np.random.choice(data_features.index, subset_count, replace=False)
    return data_features.iloc[random_indices], data_labels.iloc[random_indices]


def partition_data(data_features, data_labels, test_portion=0.4, validation_portion=0.5):

    features_train, features_temp, labels_train, labels_temp = train_test_split(data_features, data_labels,
                                                                                test_size=test_portion, random_state=42)
    features_val, features_test, labels_val, labels_test = train_test_split(features_temp, labels_temp,
                                                                            test_size=validation_portion,
                                                                            random_state=42)
    return features_train, features_val, features_test, labels_train, labels_val, labels_test


def execute_svm_training(features_train, labels_train, features_test, labels_test, kernel_type, class_names):

    print(f"Training SVM with {kernel_type} kernel... Kindly, wait a little.")
    svm_model = SVC(kernel=kernel_type, random_state=42)
    svm_model.fit(features_train, labels_train)

    print(f"Evaluating SVM with {kernel_type} kernel...")
    predicted_labels = svm_model.predict(features_test)
    confusion_mat = confusion_matrix(labels_test, predicted_labels)
    f1_result = f1_score(labels_test, predicted_labels, average='weighted')

    print(f"Confusion Matrix ({kernel_type.capitalize()} Kernel):\n", confusion_mat)
    print(f"F1-Score ({kernel_type.capitalize()} Kernel):", f1_result)

    display_confusion_matrix(confusion_mat, class_names, f"Confusion Matrix - {kernel_type.capitalize()} Kernel")


def display_confusion_matrix(confusion_mat, class_names, graph_title):

    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(graph_title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def svm_experiment_controller(preprocessed_features, target_labels):

    try:
        print("Selecting a subset of samples...")
        subset_features, subset_labels = select_random_subset(preprocessed_features, target_labels)

        print("Splitting the subset into training, validation, and testing sets...")
        features_train, features_val, features_test, labels_train, labels_val, labels_test = partition_data(
            subset_features, subset_labels)

        class_names = [chr(i + 65) for i in range(26)]  # Class labels A-Z

        # Linear Kernel
        execute_svm_training(features_train, labels_train, features_test, labels_test, kernel_type='linear',
                             class_names=class_names)

        # Nonlinear Kernel (RBF)
        execute_svm_training(features_train, labels_train, features_test, labels_test, kernel_type='rbf',
                             class_names=class_names)

    except Exception as error:
        print(f"An issue occurred while running the SVM experiment: {error}")
    # Logistic Regression Class (Implemented from Scratch)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.theta = None

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.max_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        probabilities = self.sigmoid(z)
        return np.round(probabilities)  # Binary classification (0 or 1)

    # Experiment 2: Implementing Logistic Regression


def experiment_logistic_regression(normalized_features, labels):

    try:
        subset_features, subset_labels = select_subset(normalized_features, labels)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(subset_features, subset_labels)
        X_train, X_val, X_test = add_bias_term(X_train, X_val, X_test)
        classifiers = train_logistic_regression(X_train, y_train)
        y_val_pred, y_test_pred = predict_labels(classifiers, X_val, X_test)
        evaluate_and_plot(y_val, y_val_pred, y_test, y_test_pred, len(classifiers))
    except Exception as e:
        print(f"Error during Logistic Regression experiment: {e}")

def select_subset(normalized_features, labels):
    print("Selecting a subset of 10,000 samples...")
    subset_indices = np.random.choice(normalized_features.index, 10000, replace=False)
    subset_features = normalized_features.iloc[subset_indices]
    subset_labels = labels.iloc[subset_indices]
    return subset_features, subset_labels

def split_data(subset_features, subset_labels):
    print("Splitting the subset into training, validation, and testing sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(subset_features, subset_labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def add_bias_term(X_train, X_val, X_test):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    return X_train, X_val, X_test

def train_logistic_regression(X_train, y_train):
    num_classes = len(np.unique(y_train))
    y_train_onehot = pd.get_dummies(y_train).to_numpy()
    classifiers = []
    for i in range(num_classes):
        print(f"Training logistic regression for class {chr(i + 65)}...")
        clf = LogisticRegression(learning_rate=0.1, max_iter=500)
        clf.fit(X_train, y_train_onehot[:, i])
        classifiers.append(clf)
    return classifiers

def predict_labels(classifiers, X_val, X_test):
    print("Predicting validation set labels...")
    val_probabilities = np.array([clf.sigmoid(np.dot(X_val, clf.theta)) for clf in classifiers]).T
    y_val_pred = np.argmax(val_probabilities, axis=1)

    print("Predicting test set labels...")
    test_probabilities = np.array([clf.sigmoid(np.dot(X_test, clf.theta)) for clf in classifiers]).T
    y_test_pred = np.argmax(test_probabilities, axis=1)

    return y_val_pred, y_test_pred

def evaluate_and_plot(y_val, y_val_pred, y_test, y_test_pred, num_classes):
    print("Evaluating Logistic Regression on Validation Set...")
    cm_val = confusion_matrix(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred, average='weighted')
    print("Confusion Matrix (Validation Set):\n", cm_val)
    print("F1-Score (Validation Set):", f1_val)

    print("Evaluating Logistic Regression on Test Set...")
    cm_test = confusion_matrix(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    print("Confusion Matrix (Test Set):\n", cm_test)
    print("F1-Score (Test Set):", f1_test)

    sns.heatmap(cm_val, annot=True, fmt='d', cmap='coolwarm', xticklabels=[chr(i + 65) for i in range(num_classes)],
                yticklabels=[chr(i + 65) for i in range(num_classes)])
    plt.title("Confusion Matrix - Validation Set (Logistic Regression)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm', xticklabels=[chr(i + 65) for i in range(num_classes)],
                yticklabels=[chr(i + 65) for i in range(num_classes)])
    plt.title("Confusion Matrix - Test Set (Logistic Regression)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def split_and_one_hot_encode(normalized_features, labels):

    try:

        print("Selecting a subset of 10,000 samples...")
        # Starts by taking only 10k samples from the dataset
        subset_indices = np.random.choice(normalized_features.index, 10000, replace=False)
        # Then takes the number of features to be the input neurons
        subset_features = normalized_features.iloc[subset_indices]
        # Takes the labels provided to check if the output was right
        subset_labels = labels.iloc[subset_indices]

        # Splits the dataset we got into training, validation, and testing sets
        print("Splitting the subset into training, validation, and testing sets...")
        # Takes 60% as training and 40% as temp
        X_train, X_temp, y_train, y_temp = train_test_split(subset_features, subset_labels, test_size=0.4, random_state=42)
        #Splits the 40% of the temp as 20 % validation and 20 % testing set
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Convert labels to one-hot encoding for each set , this encoding changes the labels to binary vectors
        #The vector size is based on the number of classes , since we have 26 alphabets so the size is 26
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=26)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=26)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=26)

        return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot

    except Exception as e:
        print(f"Error during data splitting and encoding: {e}")


def train_neural_network_1(X_train, y_train_onehot, X_val, y_val_onehot):
   # First Neural Network has 2 hidden layers and relu activation function
    try:
        print("Training Neural Network 1...")
        model_1 = Sequential([

            # Takes input neurons based on features
            Flatten(input_shape=(X_train.shape[1],)),
            # First hidden layer's number of neurons along with its activation function
            Dense(128, activation='relu'),
            # Drops 20% of the neurons to avoid overfitting
            Dropout(0.2),
            # Second hidden layer's number of neurons along with its activation function
            Dense(64, activation='relu'),
            # Softmax function to give the probability of each alphabet
            Dense(26, activation='softmax')
        ])

        model_1.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        #Generates 20 epochs it does this cycle for 20 times , taking 64 by 64 and patience of 3 as if it haven't improved for 3 iterations it stops
        history_1 = model_1.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot), epochs=20, batch_size=64, verbose=1,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

        # Plots Neural Network 1 performance
        plot_training_curves(history_1, "Neural Network 1")
        return model_1, history_1

    except Exception as e:
        print(f"Error during Neural Network 1 training: {e}")


def train_neural_network_2(X_train, y_train_onehot, X_val, y_val_onehot):
    # Second Neural Network has 3 hidden layers and tanh activation function
    try:
        print("Training Neural Network 2...")
        model_2 = Sequential([
            Flatten(input_shape=(X_train.shape[1],)),
            # First hidden layer's number of neurons along with its activation function
            Dense(256, activation='tanh'),
            # Drops 30% of the neurons to avoid overfitting
            Dropout(0.3),
            # Second hidden layer's number of neurons along with its activation function
            Dense(128, activation='tanh'),
            # Third hidden layer's number of neurons along with its activation function
            Dense(64, activation='tanh'),
            # Softmax function to give the probability of each alphabet
            Dense(26, activation='softmax')
        ])
        model_2.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        #Generates 20 epochs it does this cycle for 20 times , taking 64 by 64 and patience of 3 as if it haven't improved for 3 iterations it stops
        history_2 = model_2.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot), epochs=20, batch_size=64, verbose=1,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

        # Plots Neural Network 2 performance
        plot_training_curves(history_2, "Neural Network 2")
        return model_2, history_2

    except Exception as e:
        print(f"Error during Neural Network 2 training: {e}")



#Function to evaluate the model , test it , and print out a classification report
def evaluate_model(model, X_test, y_test, model_name):

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print(f"{model_name} - Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[chr(i + 65) for i in range(26)]))

#Plots the accuracy and loss curves
def plot_training_curves(history, model_name):

    plt.figure(figsize=(14, 6))

    # Plots the Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plots the Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

#Experiment 3 2 Neural Network models
def experiment_neural_networks(normalized_features, labels):

    try:
        # Step 1: Split the data and apply one-hot encoding
        X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot = split_and_one_hot_encode(normalized_features, labels)

        # Step 2: Train Neural Network 1
        model_1, history_1 = train_neural_network_1(X_train, y_train_onehot, X_val, y_val_onehot)

        # Step 3: Train Neural Network 2
        model_2, history_2 = train_neural_network_2(X_train, y_train_onehot, X_val, y_val_onehot)

        # Step 4: Evaluate on Test Set
        print("Evaluating Neural Networks on the Test Set...")
        evaluate_model(model_1, X_test, y_test_onehot, "Neural Network 1")
        evaluate_model(model_2, X_test, y_test_onehot, "Neural Network 2")

    except Exception as e:
        print(f"Error during Neural Network experiment: {e}")



# Main function
def main():
    # Load the dataset
    dataset_path = r"C:\Users\hazem\Downloads\archive\A_Z Handwritten Data.csv"
    dataset = load_dataset(dataset_path)

    # Check for column alignment issues
    if dataset is not None and dataset.shape[1] != 785:
        print(f"Dataset has {dataset.shape[1]} columns instead of 785. Please verify the file format.")
        return

    # Explore the dataset
    explore_dataset(dataset)

    # Normalize the images
    normalized_features, labels = normalize_images(dataset)

    # Display sample images
    display_sample_images(normalized_features, labels)

    #  Experiment 1
    svm_experiment_controller(normalized_features, labels)

    #  Experiment 2
    experiment_logistic_regression(normalized_features, labels)

    #  Experiment 3
    experiment_neural_networks(normalized_features, labels)


if __name__ == "__main__":
    main()