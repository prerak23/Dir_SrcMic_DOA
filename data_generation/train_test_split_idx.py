import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import random

no_rooms = 40000
tmp_ = []
for i in range(
    no_rooms
):  # It starts from room 1 and goes until room 19999, so basically train-test split misses two room's : room_0, room_20000
    for j in range(
        1
    ):  # We change value here to include more than one sources in the training dataset.
        tmp_.append(("room_" + str(i), j + 2))


print(len(tmp_))
no_of_source_to_inlude = 1

total_samples = no_rooms * no_of_source_to_inlude


train, val = (
    (total_samples * 95) / 100,
    (total_samples * 5) / 100,
)  # ,(total_samples*10)/100


random.shuffle(tmp_)

train_ar = tmp_[: int(train)]
print(len(train_ar), train_ar[0], train_ar[-1])

val_ar = tmp_[int(train) : int(train) + int(val)]
print(len(val_ar), val_ar[0], val_ar[-1])

"""
test_ar=tmp_[int(train)+int(val):int(train)+int(val)+int(test)]
print(len(test_ar),test_ar[0],test_ar[-1])3
"""
# print(train_ar)
# print(val_ar)
np.save("train_random_ar_only_source_2_included.npy", train_ar, allow_pickle=True)
np.save("val_random_ar_only_source_2_included.npy", val_ar, allow_pickle=True)
# np.save("test_random_ar.npy",test_ar,allow_pickle=True)
