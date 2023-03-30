cmdargs=$1

# `gpu=$1`
# `echo "export CUDA_VISIBLE_DEVICES=${gpu}"`
#export CUDA_VISIBLE_DEVICES='0,1'
export CUDA_VISIBLE_DEVICES='1'
hyperparameters01='[{
    "random_seed" : [4],

    "dataset" : ["cinic10"],
    "models" : [{"ConvNet" : 80}],

    "attack_rate" :  [0],
    "attack_method": ["-"],
    "participation_rate" : [0.4],

    "alpha" : [0.01],
    "eta" : [0.8],
    "client_mode": ["normal"],
    "minimum_trajectory_length": [[25]],
    "maximum_distill_round": [1],
    "distill_interval": [1],
    "start_round": [0],
    "communication_rounds" : [200],
    "local_epochs" : [1],
    "batch_size" : [32],
    "val_size" : [32],
    "val_batch_size": [32],
    "local_optimizer" : [ ["Adam", {"lr": 0.001}]],
    "distill_iter": [20],
    "distill_lr": [0.00025],

    "aggregation_mode" : ["datadistill"],

    "sample_size": [0],
    "save_scores" : [false],

    "pretrained" : [null],
    "save_model" : [null],
    "log_frequency" : [1],
    "log_path" : ["new_noniid/"]}]

'


RESULTS_PATH="results/"
DATA_PATH="../data/"
CHECKPOINT_PATH="checkpoints/"

python -u run_end2end.py --hp="$hyperparameters01" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs --dataset=cifar10 --ipc=15 --syn_steps=20 --expert_epochs=3 --max_start_epoch=50 --min_start_epoch=0 --lr_img=5e-2 --lr_lr=1e-05 --lr_teacher=0.01 --pix_init noise --img_optim adam --lr_teacher 0.04 --weight_averaging --least_ave_num 2 --start_learning_label 0 --label_init 0. --Iteration 3000 --project dynafed_cinic10 --runs_name hyperparameters01
