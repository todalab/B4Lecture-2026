"""Train anomaly classifier C.N.N., and run a quick test after."""

from cnn import predict, train

if __name__ == "__main__":
    train(["data/dev", "data/dev/abundant"])

    # Tiny test
    print(predict("data/dev/model_00_anomaly_00000108.wav"))
    print(predict("data/dev/model_06_normal_00000030.wav"))
