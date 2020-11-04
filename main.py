from config import ARGS
import util
from dataset.dataset_user_sep import UserSepDataset
from network.DKT import DKT
from network.DKVMN import DKVMN
from network.NPA import NPA
from network.SAKT import SAKT
from constant import QUESTION_NUM
from trainer import Trainer
import numpy as np
import pickle as pkl
import pandas as pd


def get_model():
    if ARGS.model == 'DKT':
        model = DKT(ARGS.input_dim, ARGS.hidden_dim, ARGS.num_layers, QUESTION_NUM[ARGS.dataset_name],
                    ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    elif ARGS.model == 'DKVMN':
        model = DKVMN(ARGS.key_dim, ARGS.value_dim, ARGS.summary_dim, QUESTION_NUM[ARGS.dataset_name],
                      ARGS.concept_num).to(ARGS.device)
        d_model = ARGS.value_dim

    elif ARGS.model == 'NPA':
        model = NPA(ARGS.input_dim, ARGS.hidden_dim, ARGS.attention_dim, ARGS.fc_dim,
                    ARGS.num_layers, QUESTION_NUM[ARGS.dataset_name], ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    elif ARGS.model == 'SAKT':
        model = SAKT(ARGS.hidden_dim, QUESTION_NUM[ARGS.dataset_name], ARGS.num_layers,
                     ARGS.num_head, ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    else:
        raise NotImplementedError

    return model, d_model


test_sample_infos = None
num_of_test_user = None
def run(i):
    """
    i: single integer represents dataset number
    """
    global test_sample_infos
    global num_of_test_user

    user_base_path = f'{ARGS.base_path}/{ARGS.dataset_name}/processed'
    print(ARGS)

    train_data_path = f'{user_base_path}/{i}/train/'
    val_data_path = f'{user_base_path}/{i}/val/'
    test_data_path = f'{user_base_path}/{i}/test/'

    train_sample_infos, num_of_train_user = util.get_data_user_sep(train_data_path)
    val_sample_infos, num_of_val_user = util.get_data_user_sep(val_data_path)
    if test_sample_infos is None:
        test_sample_infos, num_of_test_user = util.get_data_user_sep(test_data_path)

    train_data = UserSepDataset('train', train_sample_infos, ARGS.dataset_name)
    val_data = UserSepDataset('val', val_sample_infos, ARGS.dataset_name)
    test_data = UserSepDataset('test', test_sample_infos, ARGS.dataset_name)

    print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_infos)}')
    print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_infos)}')
    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

    model, d_model = get_model()

    trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
                      d_model, ARGS.num_epochs, ARGS.weight_path,
                      ARGS.lr, train_data, val_data, test_data)
    trainer.train()
    trainer.test(0)
    return trainer.test_acc, trainer.test_auc

def predict():
    test_data_path = f'{ARGS.base_path}/{ARGS.dataset_name}/data_test/'

    print(test_data_path)

    user_infos, test_sample_infos, num_of_test_user = util.get_data_user_sep_test(test_data_path)

    test_data = UserSepDataset('test', test_sample_infos, ARGS.dataset_name)

    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_infos)}')

    model, d_model = get_model()

    trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
                      d_model, ARGS.num_epochs, ARGS.weight_path,
                      ARGS.lr, None, None, test_data)
    predictions = trainer.predict(25000)
    # pkl.dump(predictions, open('predictions.pkl', 'wb'))

    assert len(predictions) == len(user_infos), \
        "Invalid num of user infos and num of predictions, {} vs {}".format(len(predictions), len(user_infos))
    submission = {}
    submission['row_id'] = []
    submission['answered_correctly'] = []
    submission['groups'] = []
    for idx, pre in enumerate(predictions):
        submission['row_id'].append(user_infos[idx]['row_id'])
        submission['answered_correctly'].append(pre)
        submission['groups'].append(user_infos[idx]['group_id'])

    pd.DataFrame.from_dict(submission).to_csv('../submission.csv', index=False)
    # print(submission)


if __name__ == '__main__':
    if ARGS.prediction:
        predict()
    else:
        if ARGS.cross_validation is False:
            test_acc, test_auc = run(1)
        else:
            acc_list = []
            auc_list = []

            for i in range(1, 10):
                print(f'{i}th dataset')
                test_acc, test_auc = run(i)
                acc_list.append(test_acc)
                auc_list.append(test_auc)

            acc_array = np.asarray(acc_list)
            auc_array = np.asarray(auc_list)
            print(f'mean acc: {np.mean(acc_array):.4f}, auc: {np.mean(auc_array):.4f}')
            print(f'std acc: {np.std(acc_array):.4f}, auc: {np.std(auc_array):.4f}')
