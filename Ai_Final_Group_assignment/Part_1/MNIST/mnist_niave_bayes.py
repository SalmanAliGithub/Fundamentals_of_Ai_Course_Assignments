from mnist import MNIST
import math
import matplotlib.pyplot as plt

# Load MNIST data
mnist_data = MNIST('mnistData')
mnist_data.gz = True

train_images, train_labels = mnist_data.load_training()
num_images, num_pixels = len(train_images), len(train_images[0])

# Laplace smoothing parameter
laplace_smoothing = 2

# Frequency distributions
pixel_freq_dist = [[laplace_smoothing for _ in range(10)] for _ in range(num_pixels)]
total_freq = [laplace_smoothing for _ in range(10)]

def train_naive_bayes():
    print("Training Naive Bayes...")
    global pixel_freq_dist, total_freq

    pixel_freq_dist = [[laplace_smoothing for _ in range(10)] for _ in range(num_pixels)]
    total_freq = [laplace_smoothing * num_pixels * 255 for _ in range(10)]

    for i in range(num_images):
        image = train_images[i]

        for px in range(num_pixels):
            if image[px]:
                pixel_freq_dist[px][train_labels[i]] += image[px]

        total_freq[train_labels[i]] += 255

    for digit in range(10):
        for px in range(num_pixels):
            pixel_freq_dist[px][digit] /= total_freq[digit]

def predict_naive_bayes(image):
    probabilities = [0 for _ in range(10)]

    for px in range(num_pixels):
        if image[px] > 160:
            for digit in range(10):
                probabilities[digit] += math.log(pixel_freq_dist[px][digit])
        else:
            for digit in range(10):
                probabilities[digit] += math.log(1 - pixel_freq_dist[px][digit])

    return probabilities.index(max(probabilities))

nb_results = []

def test_naive_bayes():
    print("Testing Naive Bayes...")
    test_images, test_labels = mnist_data.load_testing()
    num_tests = 100
    correct_predictions = 0

    for i in range(num_tests):
        predicted = predict_naive_bayes(test_images[i])
        actual = test_labels[i]

        if predicted == actual:
            correct_predictions += 1

    nb_results.append(correct_predictions / num_tests)
    print("Accuracy: " + str(round((correct_predictions / num_tests) * 100, 2)))


laplace_values = [0.1, 0.5, 1, 2, 10, 50, 100, 500]

for lv in laplace_values:
    laplace_smoothing = lv
    train_naive_bayes()
    print("Laplace", lv)
    test_naive_bayes()
    print("")

plt.plot(laplace_values, nb_results)
plt.xlabel("Laplace Smoothing")
plt.ylabel("Accuracy")
plt.title("Laplace Smoothing vs Accuracy for The MNIST Digit Dataset")
plt.show()