# DKT
# python main.py --num_workers=4 --gpu=0 --device=cuda --model=DKT --num_epochs=6  \
# --eval_steps=5000 --train_batch=2048 --test_batch=2048 --seq_size=200 \
# --input_dim=100 --hidden_dim=100 --name=Riid_DKT_dim_100_100 \
# --dataset_name=riiid --cross_validation=1

## DKVMN
#python main.py --num_workers=8 --gpu=0 --device=cuda --model=DKVMN --num_epochs=6
#--eval_steps=5000 --train_batch=2048 --test_batch=2048 --seq_size=200 --concept_num=50 --batch_size=256
#--input_dim=100 --hidden_dim=100 --key_dim=100 --value_dim=100 --summary_dim=50 --name=EdNet-KT1_DKVMN_dim_100_100
#--dataset_name=EdNet-KT1 --cross_validation=1
#
## NPA
python main.py --num_workers=1 --gpu=0 --device=cuda --model=NPA --num_epochs=6 --attention_dim=100 --fc_dim=200 \
--eval_steps=5000 --train_batch=1024 --test_batch=1024 --seq_size=200 \
--input_dim=100 --hidden_dim=100 --name=riiid_NPA_dim_100_100 \
--dataset_name=riiid --cross_validation=1 --prediction
#
## SAKT
#python main.py --num_workers=8 --gpu=0 --device=cuda --model=DKT --num_epochs=6
#--eval_steps=5000 --train_batch=512 --test_batch=2048 --seq_size=200
#--input_dim=100 --hidden_dim=200 --name=EdNet-KT1_SAKT_dim_100_100
#--dataset_name=EdNet-KT1 --cross_validation=1
