import os
import matplotlib.pyplot as plt
import seaborn as sns

filenames = [x for x in os.listdir() if ".txt" in x]
TIME = 0
SITE = 1
TRAIN_LOSS = 2
VAL_LOSS = 3
CUR_PATIENCE = 4
BEST_LOSS = 5
CUR_EPOCH = 6

for i in range(len(filenames)):
    epochs = []
    train_losses = []
    val_losses = []
    with open(filenames[0], 'r') as f:
        iter_lines = iter(f.readlines())
        next(iter_lines)
        for line in iter_lines:
            cur_data = line.split()

            epochs.append(int(cur_data[CUR_EPOCH]))
            train_losses.append(float(cur_data[TRAIN_LOSS]))
            val_losses.append(float(cur_data[VAL_LOSS]))

    ax = sns.lineplot(x=epochs,
                      y=val_losses,
                      color=sns.xkcd_rgb["light orange"],
                      label="Validation Loss")
    ax = sns.lineplot(x=epochs,
                      y=train_losses,
                      color=sns.xkcd_rgb["cerulean"],
                      label="Train Loss")

    ax.set(xlabel="Epoch",
           ylabel="Loss")
    ax.legend()

    plt.title("Model Loss by Epoch")
    plt.show()
