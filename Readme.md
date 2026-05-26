SAOT: Self-Supervised Continual Graph Learning with Structure-Aware Optimal Transport

###Representation Learning
python train.py --dataset CoraFull-CL --gpu 0 --ot_struct_lambda 0.5 --distill_lambda 0.6 --enable_distill True 

###Linear evaluation
python evaluate.py --dataset Corafull-CL  --scenario class  --backbone GCN
