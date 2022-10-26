set -e
modality=$1
run_idx=$2
gpu=$3

for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal_miss --model=MISA
--gpu_ids=$gpu --modality=$modality --corpus_name=IEMOCAP
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10
--A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --embd_size_l=128
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
--niter=20 --niter_decay=20 --verbose --beta1=0.9
--batch_size=64 --lr=2e-4 --run_idx=$run_idx
--name=_MISA_missing --suffix=_run_{run_idx} --has_test
--cvNo=$i --num_classes=4 --hidden_size=384
--diff_weight=0.06 --sim_weight=0.2 --recon_weight=0.2"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done