import matplotlib.pyplot as plt
import numpy as np


colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # Single-letter colors
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 
    'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 
    'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 
    'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 
    'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 
    'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 
    'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 
    'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 
    'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 
    'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 
    'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 
    'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 
    'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 
    'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 
    'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 
    'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 
    'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 
    'rosybrown', 'royal']

def plot_sine_approximation(axes, x_values, X_train, y_train, fitted_model, model_on_training_data, X_val, y_val, plot_title):
    
    axes.plot(x_values[:, 0], np.sin(x_values[:, 0]), label="True sin(x)")
    axes.plot(x_values[:, 0], fitted_model, label="Fitted model for sin(x)", linestyle="dashed", color='pink')

    axes.plot(X_train[:, 0], y_train , label="Training data", marker='o', linestyle='', color='blue')
    axes.scatter(X_train[:, 0], model_on_training_data, label="Fitted model on training data", linestyle="dashed", color='red')

    axes.plot(X_val[:, 0], y_val, label="Validation data", marker="v", color='green', linestyle='')    


    axes.set_title(f"{plot_title}")
    axes.legend(loc='lower right', ncol=2)

    return axes


def plot_loss(axes, train_loss, val_loss, epoch_number, nodes):
    axes[1].plot(train_loss, label="Training loss", linestyle="dashed")
    axes[1].plot(val_loss, label="Validation loss", linestyle="dashed", color='green')
    axes[1].set_title(f"Loss for epoch:{epoch_number} - nodes:{nodes}")
    #axes[1].set_yscale('log', base=2)
    axes[1].legend()


def plot_weights_first_input(axes, all_weights, nodes, epoch_number, order):
    for neuron in range(nodes):
        axes[2].plot(all_weights['0_layer'][:, neuron, 0], label=f"Weight of the {neuron} neuron, first input", color=colors[neuron])

    axes[2].set_title(f"Weights first input: epoch:{epoch_number}  - order:{order}")
    axes[2].axhline(y=0, color='red', linestyle='--', label='weight=0')


def plot_weights_second_input(axes, all_weights, nodes, epoch_number, order):
    for neuron in range(nodes):
        axes[3].plot(all_weights['0_layer'][:, neuron, 1], label=f"Weight of the {neuron} neuron, second input", color=colors[neuron], linestyle='--')

    axes[3].set_title(f"Weights second input: epoch:{epoch_number}  - order:{order}")
    axes[3].axhline(y=0, color='red', linestyle='--', label='weight=0')


def my_plot(
    axes,
    x_values,
    training_X,
    training_y,
    predictions,
    predictions_on_training,
    validation,
    validation_y,
    weights_b1,
    all_weights,
    train_loss,
    val_loss,
    nodes,
    epoch_number,
    order,
):
    plot_sine_approximation(axes, x_values, training_X, training_y, predictions, predictions_on_training, validation, validation_y, nodes, epoch_number)
    plot_loss(axes, train_loss, val_loss, epoch_number, nodes)
    plot_weights_first_input(axes, all_weights, nodes, epoch_number, order)
    plot_weights_second_input(axes, all_weights, nodes, epoch_number, order)

    plt.tight_layout()
    plt.show()
