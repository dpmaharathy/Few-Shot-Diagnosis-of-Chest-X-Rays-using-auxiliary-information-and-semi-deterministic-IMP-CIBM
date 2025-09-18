# Few-Shot-Diagnosis-of-Chest-X-Rays-using-auxiliary-information-and-semi-deterministic-IMP-CIBM
This repository contains the code for the paper "Few-shot diagnosis of chest x-ray images using auxiliary information guided semi-deterministic infinite mixture prototypes" accepted in the journal Computers in Biology &amp; Medicine (2025).


In order to execute the code of proposed method just navigate into the codes directory and then run the following command.

```
python chexpert_proposed_128_1.py > logfile.txt
```

To change the dataset split change the path of x and y for each training, testing and validation as follows:

``
path_x = "/dir/chexpert_128/X_val_3.npy"
path_y = "/dir/chexpert_128/y_val_3.npy"
```
to 

```
path_x = "/dir/chexpert_128/X_val_4.npy"
path_y = "/dir/chexpert_128/y_val_4.npy"
```
If you want to perform experiments on split 4 instead of split 3.
Also change the path for model saving according in this line of code.

```
path = "/dir/models/split5/"

```
The model configuration can be changed in the for loop as follows:
```
    n_way = 8
    n_support = 5
    n_query = 5

    n_way_val = 3
    n_support_val = 5
    n_query_val = 1

    train_x = Xtrain
    train_y = ytrain

    val_x=Xval
    val_y=yval

    max_epoch = 2
    epoch_size = 500
    temp_str = path + 'chexpert_proposed_1_128*128_' + str(i)
```

You can get the data files (numpy) from this link https://www.kaggle.com/datasets/dpmaharathy/nih-chexpert-split-dataset
