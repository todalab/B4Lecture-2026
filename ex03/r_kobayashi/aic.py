import sys

import main2
import matplotlib.pyplot as plt
import numpy as np


def main(arg):
    arg = str(arg)
    AIC_list = []
    BIC_list = []

    if arg == "1" or arg == "2" or arg == "3":
        for k in range(1, 7):
            log_list, N, p = main2.main(arg, k)
            logL = log_list[-1]

            AIC = -2 * logL + 2 * p
            BIC = -2 * logL + p * np.log(N)
            AIC_list.append(AIC)
            BIC_list.append(BIC)

            print("AIC:", AIC)
            print("BIC:", BIC)

        plt.title("AIC & BIC")
        plt.xlabel("k", fontsize=18)
        plt.ylabel("AIC & BIC", fontsize=18)
        plt.grid(which="both")
        AIC_min_index = AIC_list.index(min(AIC_list)) + 1
        BIC_min_index = BIC_list.index(min(BIC_list)) + 1
        plt.vlines(
            [AIC_min_index],
            min(AIC_list),
            max(AIC_list),
            color="red",
            linestyles="dashed",
            linewidth=1,
        )
        plt.vlines(
            [BIC_min_index],
            min(BIC_list),
            max(BIC_list),
            color="blue",
            linestyles="dashed",
            linewidth=1,
        )
        plt.plot(range(1, 7), AIC_list, label="AIC")
        plt.plot(range(1, 7), BIC_list, label="BIC")
        plt.legend(fontsize=8)
        plt.savefig("AICBIC_data" + arg)
        plt.show()

    else:
        print(f"不正な引数です: {arg}。1, 2, 3 のいずれかを指定してください。")
        sys.exit(1)


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        print("引数が必要です")
        sys.exit(1)

    main(arg)
