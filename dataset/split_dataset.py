import os
import sys
import random
from tqdm import tqdm

data_path = "KT_benchmark_dataset/riiid"
n_splits = 5

# trains = os.listdir(os.path.join(data_path, "_1/train"))
# vals = os.listdir(os.path.join(data_path, "_1/val"))
# tests = os.listdir(os.path.join(data_path, "_1/test"))

users = os.listdir("KT_benchmark_dataset/riiid/data")
n_trains = int(len(users) * 0.85)
n_vals = (len(users) - n_trains) // 10 * 9

trains = random.sample(users, k=n_trains)
users = list(set(users) - set(trains))
vals = random.sample(users, k=n_vals)
users = list(set(users) - set(vals))
tests = list(set(users) - set(trains) - set(vals))

users = os.listdir("KT_benchmark_dataset/riiid/data")
print("Total samples #{} \t Training samples #{} \t Validation samples #{} \t Testing samples #{}".format(len(users), len(trains), len(vals), len(tests)))




n_train_user = len(trains) // n_splits
n_val_user = len(vals) // n_splits

n_train_user = 40000
n_val_user = int( n_train_user * len(vals) / len(trains))
n_splits = len(trains) // n_train_user + 1

for i in tqdm(range(n_splits), desc="Splitting data..."):
    os.mkdir(os.path.join(data_path, "processed/{}/".format(i + 1)))
    os.mkdir(os.path.join(data_path, "processed/{}/train".format(i + 1)))
    os.mkdir(os.path.join(data_path, "processed/{}/val".format(i + 1)))
    # if os.path.isdir(os.path.join(data_path, "processed/{}/test".format(i + 1))) is False:
    #     os.mkdir(os.path.join(data_path, "processed/{}/test".format(i + 1)))
    if i < n_splits - 1:
        train_piece = random.sample(trains, k=n_train_user)
        val_piece = random.sample(vals, k=n_val_user)
    else:
        train_piece = trains
        val_piece = vals
    for user in train_piece:
        os.system("cp {} {}".format(os.path.join(data_path, "data/{}".format(user)),
                                    os.path.join(data_path, "processed/{}/train/{}".format(i + 1, user))))
    for user in val_piece:
        os.system("cp {} {}".format(os.path.join(data_path, "data/{}".format(user)),
                                    os.path.join(data_path, "processed/{}/val/{}".format(i + 1, user))))
    if i == 0:
        for user in tests:
            os.system("cp {} {}".format(os.path.join(data_path, "data/{}".format(user)),
                                        os.path.join(data_path, "test")))

    # for user in tests:
    os.system("ln -s {} {}".format(os.path.join("/home/ubuntu/papers/DKT/KT/dataset", data_path, "test"),
                                   os.path.join("/home/ubuntu/papers/DKT/KT/dataset", data_path, "processed/{}".format(i + 1))))

    trains = list(set(trains) - set(train_piece))
    vals = list(set(vals) - set(val_piece))

    # if i < n_splits - 1:
    #     tests_piece = random.sample(tests, k=n_test_user)
    # else:
    #     tests_piece = tests
    # #     val_piece = vals
    # for user in tests_piece:
    #     os.system("cp {} {}".format(os.path.join(data_path, "_1/test/{}".format(user)),
    #                                 os.path.join(data_path, "processed/{}/test/{}".format(i + 1, user))))

