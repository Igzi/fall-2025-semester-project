source /home/ubuntu/LoraRetriever/myenv/bin/activate
python3 main.py --data_path  dataset/combined_test.json  --res_path results/mixture.json --eval_type mixture  --lora_num 3 --batch_size 1
python3 main.py --data_path  dataset/combined_test.json  --res_path results/mixture_ood.json --eval_type mixture  --lora_num 3 --batch_size 1 --ood=True
python3 main.py --data_path  dataset/combined_test.json  --res_path results/fusion.json --eval_type fusion  --lora_num 3 --batch_size 1
python3 main.py --data_path  dataset/combined_test.json  --res_path results/fusion_ood.json --eval_type fusion  --lora_num 3 --batch_size 1 --ood=True
python3 main.py --data_path  dataset/combined_test.json  --res_path results/selection.json --eval_type fusion  --lora_num 1 --batch_size 1
python3 main.py --data_path  dataset/combined_test.json  --res_path results/selection_ood.json --eval_type fusion  --lora_num 1 --batch_size 1 --ood=True
python3 main.py --data_path  dataset/combined_test.json  --res_path results/best_selection.json --eval_type mixture  --lora_num 1 --batch_size 1 --best_selection=True