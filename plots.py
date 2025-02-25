import matplotlib.pyplot as plt
import numpy as np
from utils import colors
import torch
import tempfile
from PIL import Image
import pdb


# fmt: off
def plot_sine_approximation(
    axes,
    x_values,
    fitted_model_on_x_values,
    X_train,
    y_train,
    model_on_training_data,
    X_val,
    y_val,
    model_on_validation_data,
    plot_title):
    """
    Args:
        axes: axes of the plot
        x_values [float] : list of values on the x axes (usually a linspace)
        fitted_model [float]: model(x_values)
        X_train [float, float]: training set, domain
        y_train [float]: training set, f(x)
        model_on_training_data [float]: model(X_train)
        X_val [float, float]: validation set, domain
        y_val [float]: validation set, f(x)
        plot_title "str": title of the plot
    """


    axes.plot(x_values[:, 0],  np.sin(x_values[:, 0]), label="True sin(x)")
    axes.scatter(x_values[:, 0],  fitted_model_on_x_values,       label="Fitted model", marker='o',  linestyle='-',   color="red", s=1)

    axes.scatter(X_train[:, 0], y_train, label="Training data", marker="o", linestyle="", color="blue", s=10 )
    axes.scatter( X_val[:, 0], y_val, label="Validation data", marker="v",  linestyle="",  color="green", s=10 )

    axes.scatter(X_train[:, 0], model_on_training_data, label="Fitted model on training data", marker="o", color="deepskyblue", s=10)
    axes.scatter(X_val[:, 0], model_on_validation_data, label="Fitted model on validation data", marker="v", color="lime", s=10)


    axes.set_title(f"{plot_title}")

    axes.legend(loc="lower right", ncol=2)

    return axes


def plot_loss(axes, train_loss, val_loss, epoch_number, nodes):
    axes[1].plot(train_loss, label="Training loss", linestyle="dashed")
    axes[1].plot(val_loss, label="Validation loss", linestyle="dashed", color="green")
    axes[1].set_title(f"Loss for epoch:{epoch_number} - nodes:{nodes}")
    axes[1].legend()


def plot_weights_first_input(axes, all_weights, nodes, epoch_number, order):
    for neuron in range(nodes):
        axes[2].plot(
            all_weights["0_layer"][:, neuron, 0],
            label=f"Weight of the {neuron} neuron, first input",
            color=colors[neuron],
        )

    axes[2].set_title(f"Weights first input: epoch:{epoch_number}  - order:{order}")
    axes[2].axhline(y=0, color="red", linestyle="--", label="weight=0")


def plot_weights_second_input(axes, all_weights, nodes, epoch_number, order):
    for neuron in range(nodes):
        axes[3].plot(
            all_weights["0_layer"][:, neuron, 1],
            label=f"Weight of the {neuron} neuron, second input",
            color=colors[neuron],
            linestyle="--",
        )

    axes[3].set_title(f"Weights second input: epoch:{epoch_number}  - order:{order}")
    axes[3].axhline(y=0, color="red", linestyle="--", label="weight=0")


# def my_plot(axes, x_values, X_train, y_train, model_on_validation_data,
#             model_on_training_data,
#             X_val, y_val,
#             weights_b1, all_weights,
#             train_loss, val_loss, nodes, epoch_number, order):

#     # Fitted model predictions_on_training, # predictions on training data validation, validation_y, nodes, # filename )
#     plot_sine_approximation(axes,
#                             x_values=x_values,
#                             fitted_model_on_x_values=y_values,
#                             X_train=X_train,
#                             y_train=y_train,
#                             model_on_training_data=model_on_training_data,
#                             X_val=X_val,
#                             y_val=y_val,
#                             model_on_validation_data=model_on_validation_data,
#                             plot_title="Sin(x) approximation",)
#     # x_values,
#     # fitted_model_on_x_values,
#     # X_train,
#     # y_train,
#     # model_on_training_data,
#     # X_val,
#     # y_val,
#     # plot_title="Sin(x) approximation",)


#         # axes: axes of the plot
#         # x_values [float] : list of values on the x axes (usually a linspace)
#         # fitted_model [float]: model(x_values)
#         # X_train [float, float]: training set, domain
#         # y_train [float]: training set, f(x)
#         # model_on_training_data [float]: model(X_train)
#         # X_val [float, float]: validation set, domain
#         # y_val [float]: validation set, f(x)
#         # plot_title "str": title of the plot


# plot_loss(axes, train_loss, val_loss, epoch_number, nodes)
# plot_weights_first_input(axes, all_weights, nodes, epoch_number, order)
# plot_weights_second_input(axes, all_weights, nodes, epoch_number, order)

# # plt.tight_layout()
# # plt.show()


# fmt: on
def plot_all_weights(model, weights_at_layer):
    """
    Plots the evolution of the first layer input weights over training iterations,
    then saves and logs the image as an artifact.
    """
    # pdb.set_trace()
    if len(weights_at_layer) == 0:
        print("No weights to plot.")
        return

    # for every layer in the model create a plot
    # that shows the evolution of the weights

    # create a figure with as many subplots as layers
    fig, axes = plt.subplots(
        len(weights_at_layer),
        1,
        figsize=(6, 3 * len(weights_at_layer)),  # Make each subplot 3 inches tall
        dpi=100,
    )

    # For each layer in the model
    for i, (layer_name, layer_weights) in enumerate(weights_at_layer.items()):
        # Convert list of tensors to numpy array
        weights_array = np.array([w.numpy() for w in layer_weights])

        # Debug print
        # print(f"Layer {layer_name} weights shape: {weights_array.shape}")

        # pdb.set_trace()
        # Plot each neuron's weights
        for j in range(weights_array.shape[1]):
            axes[i].plot(weights_array[:, j, :], label=f"Neuron {j+1}")

        axes[i].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(4, weights_array.shape[1]),  # Up to 4 columns
            borderaxespad=0.0,
        )
        axes[i].set_title(f"Evolution of weights in {layer_name}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Weight value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close(fig)
        try:
            img = Image.open(tmpfile.name)
            model.logger.experiment.log_image(
                image=img,
                artifact_file="evolution_weights.png",
                run_id=model.logger.run_id,
            )
        except AttributeError:
            print("Could not log the image file as an artifact.")


def plot_firing_value(model, firings_at_layer):
    """
    Plots the evolution of the first layer input weights over training iterations,
    then saves and logs the image as an artifact.
    """
    # pdb.set_trace()
    if len(firings_at_layer) == 0:
        print("No firing value to plot.")
        return

    # for every layer in the model create a plot
    # that shows the evolution of the weights

    # create a figure with as many subplots as layers
    fig, axes = plt.subplots(len(firings_at_layer), 1, figsize=(8, 8), dpi=100)

    # For each layer in the model
    for i, (layer_name, layer_firings) in enumerate(firings_at_layer.items()):
        # Convert list of tensors to numpy array
        # pdb.set_trace()
        weights_array = np.array(layer_firings)

        # Debug print
        print(f"Layer {layer_name} weights shape: {weights_array.shape}")
        # pdb.set_trace()

        # Plot each neuron's weights
        for j in range(weights_array.shape[1]):
            axes[i].plot(weights_array[:, j], label=f"Neuron {j+1}")

        axes[i].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(4, weights_array.shape[1]),  # Up to 4 columns
            borderaxespad=0.0,
        )
        axes[i].set_title(f"Firing of neurons {layer_name} over epochs")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Firing value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close(fig)
        try:
            img = Image.open(tmpfile.name)
            model.logger.experiment.log_image(
                image=img,
                artifact_file="evolution_firing.png",
                run_id=model.logger.run_id,
            )
        except AttributeError:
            print("Could not log the image file as an artifact.")
