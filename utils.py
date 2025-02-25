import argparse
import tempfile
from PIL import Image
import matplotlib.pyplot as plt


def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lambda_reg", type=float)
    parser.add_argument("--regularization_order", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--nn_architecture", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_size", type=int)
    parser.add_argument("--validation_ratio", type=float)
    parser.add_argument("--noise_of_noisy_feature", type=float)
    parser.add_argument("--interval", type=float)
    parser.add_argument("--non_linearity", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--how", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--how_often_to_plot", type=int)
    parser.add_argument("--scheduler", type=str)
    return parser


def save_and_log_plot(fig, run_idz, epochz, loggerz):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close(fig)
        try:
            loggerz.experiment.log_image(
                image=Image.open(tmpfile.name),
                artifact_file=f"model_output_epoch_{epochz:04d}.png",
                run_id=run_idz,
            )
        except Exception as e:
            print("Could not log the image file as an artifact.", e)


colors = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",
    "w",  # Single-letter colors
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "green",
    "greenyellow",
    "grey",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royal",
]
