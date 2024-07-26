from mnist import MNIST
import math
import matplotlib.pyplot as plt

# Load MNIST data
mnist_data = MNIST('mnistData')
mnist_data.gz = True

train_images, train_labels = mnist_data.load_training()
num_images, num_pixels = len(train_images), len(train_images[0])

laplace_smoothing = 2

# Frequency distributions
row_freq_dist = [[laplace_smoothing for _ in range(10)] for _ in range(28)]
total_row_freq = [laplace_smoothing for _ in range(10)]

def featurize_image(image):
    features = []

    for row in range(28):
        row_count = 0
        for col in range(28):
            if image[row * 28 + col]:
                row_count += 1
        
        features.append(row_count)

    return features

def train_row_based_naive_bayes():
    print("Training Row-Based Naive Bayes...")
    global row_freq_dist, total_row_freq

    row_freq_dist = [[laplace_smoothing for _ in range(10)] for _ in range(28)]
    total_row_freq = [laplace_smoothing * 28 for _ in range(10)]

    for i in range(num_images):
        image = train_images[i]

        for row in range(28):
            for col in range(28):
                if image[row * 28 + col]:
                    row_freq_dist[row][train_labels[i]] += 1

        total_row_freq[train_labels[i]] += 1

    for digit in range(10):
        for row in range(28):
            row_freq_dist[row][digit] /= total_row_freq[digit]

def predict_row_based_naive_bayes(image):
    probabilities = [0 for _ in range(10)]
    features = featurize_image(image)

    for row in range(28):
        for digit in range(10):
            if abs(row_freq_dist[row][digit] - features[row]) < 5:
                probabilities[digit] += 1
    
    return probabilities.index(max(probabilities))

rb_nb_results = []

def test_row_based_naive_bayes():
    print("Testing Row-Based Naive Bayes...")
    test_images, test_labels = mnist_data.load_testing()
    num_tests = 100
    correct_predictions = 0

    for i in range(num_tests):
        predicted = predict_row_based_naive_bayes(test_images[i])
        actual = test_labels[i]

        if predicted == actual:
            correct_predictions += 1

    rb_nb_results.append(correct_predictions / num_tests)
    print("Accuracy: " + str(round((correct_predictions / num_tests) * 100, 2)))

laplace_values = [0.1, 0.5, 1, 2, 10, 50, 100, 500]

for lv in laplace_values:
    laplace_smoothing = lv
    train_row_based_naive_bayes()
    print("Laplace", lv)
    test_row_based_naive_bayes()
    print("")

plt.plot(laplace_values, rb_nb_results)
plt.xlabel("Laplace Smoothing")
plt.ylabel("Accuracy")
plt.title("Laplace Smoothing vs Accuracy for The MNIST Digit Dataset (Row-Based)")
plt.show()

train_row_based_naive_bayes()
test_row_based_naive_bayes()