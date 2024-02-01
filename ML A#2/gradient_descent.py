import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gradient_descent(x, y, theta, alpha, num_iters):
    theta_not = []
    theta_one = []
    m = len(y)
    cost = []
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = np.dot(x, theta)
        error = h - y
        gradient = np.dot(x.T, error) / m
        theta = theta - alpha * gradient
        J_history[i] = np.sum(error ** 2) / (2 * m)
        cost.append(J_history[i])
        theta_not.append(theta[0])
        theta_one.append(theta[1])
    print(error)
    return theta, J_history, theta_not, theta_one, cost

def predict(x, theta):
    return np.dot(x, theta)

def main():
    x = np.array([[1, 15], [1, 20], [1, 25], [1, 28], [1, 33], [1, 16], [1, 23], [1, 31]])
    y = np.array([1, 3, 4, 6, 11, 4, 7, 12])
    theta = np.array([0, 0])
    alpha = 0.0008
    num_iters = 10000
    theta, J_history, theta0, theta1, cost = gradient_descent(x, y, theta, alpha, num_iters)

    df = pd.DataFrame({'Theta_0': theta0, 'Theta_1': theta1, 'Cost': cost})

    # Write the dataframe into a CSV file
    df.to_csv('analysis_gradient_descent.csv', index=False)

    print("Theta found by gradient descent:", theta)
    # print(f"Cost values are: {J_history}")
    # print("Prediction:", predict(np.array([[1, 4]]), theta))

    # plot the data
    plt.figure(1)
    plt.scatter(x[:,1], y)
    # plot the hypothesis
    hypothesis = predict(x, theta)
    plt.plot(x[:,1], hypothesis)
    plt.xlabel("x")
    plt.ylabel("y")

    # plot the cost function
    plt.figure(2)
    plt.plot(np.arange(num_iters), J_history.flatten())
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

if __name__ == '__main__':
    main()
