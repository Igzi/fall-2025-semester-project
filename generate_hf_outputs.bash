python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 0
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 32
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 64
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 96
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 128
rm -rf ~/.cache/huggingface/hub
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 160
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 192
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 224
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 256
rm -rf ~/.cache/huggingface/hub
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 288
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 320
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 352
python3 performance_large/generate_hf_model_outputs.py --batch_size 32 --start_id 384
rm -rf ~/.cache/huggingface/hub
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 416
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 476
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 536
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 596
rm -rf ~/.cache/huggingface/hub
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 656
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 716
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 776
rm -rf ~/.cache/huggingface/hub
python3 performance_large/generate_hf_model_outputs.py --batch_size 60 --start_id 836
python3 performance_large/generate_hf_model_outputs.py --batch_size 90 --start_id 896
python3 performance_large/generate_hf_model_outputs.py --batch_size 90 --start_id 986
python3 performance_large/generate_hf_model_outputs.py --batch_size 90 --start_id 1076
rm -rf ~/.cache/huggingface/hub
python3 performance_large/generate_hf_model_outputs.py --batch_size 128 --start_id 1166
python3 performance_large/generate_hf_model_outputs.py --batch_size 128 --start_id 1294
python3 performance_large/generate_hf_model_outputs.py --batch_size 128 --start_id 1422
python3 performance_large/generate_hf_model_outputs.py --batch_size 128 --start_id 1550