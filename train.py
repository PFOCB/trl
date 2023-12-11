from trl import SFTTrainer

exit()
# imports
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
from trl import SFTTrainer

dataset_path = '/home/g1-s23/dev/TrainingLLM/train2.csv'
# get dataset

features = Features({'input': Value('string'), 'output': Value('string')})

dataset = load_dataset('csv', data_dir=dataset_path, column_names=['input', 'output'],
                       data_files='train3.csv', split="train", features=features,
                       delimiter=',')

model_path = "/home/g1-s23/dev/7b llam2 Model/Llama-2-7b-chat-hf"


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f'### Input: {example["input"][i]}\n### Output: {example["output"][i]}'
        output_texts.append(text)
    return output_texts


# get trainer
trainer = SFTTrainer(
    model=model_path,
    train_dataset=dataset,
    dataset_text_field="text",
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    dataset_batch_size=1,
)

# train
trainer.train()


# python examples/scripts/sft_trainer.py --model_name "/home/g1-s23/dev/7b llam2 Model/Llama-2-7b-chat-hf" --dataset_name Seif-Sallam/ShareChats-GPT --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1


# python examples/scripts/sft_trainer.py --model_name "/home/g1-s23/dev/7b llam2 Model/Llama-2-7b-chat-hf" --dataset_name Seif-Sallam/ShareChats-GPT --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --checkpoint "./output/checkpoint-61900"


# python examples/scripts/sft_trainer.py --model_name "/home/g1-s23/dev/Vicuna 7b llama2" --dataset_name "/home/g1-s23/dev/ChatsDataset" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1

# python examples/scripts/sft_trainer.py --model_name "~/vicuna-7b/Vicuna 7b llama2" --dataset_name "~/dataset/" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1

# python examples/scripts/sft_trainer.py --model_name "~/vicuna-/home/g1-s23/dev/trl2/examples/scripts/sft.pys 1 --checkpoint "./output/checkpoint-3700"

# python examples/scripts/sft_trainer.py --model_name "/home/g1-s23/dev/model weights/Vicuna 7b llama2" --dataset_name "/home/g1-s23/dev/DatasetClean/" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1 --output_dir "./output-47k-cleaned"

# python examples/scripts/sft_trainer.py --model_name "/home/g1-s23/dev/model weights/Vicuna 7b llama2" --dataset_name "/home/g1-s23/dev/DatasetClean/phase3" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1 --output_dir "./output-47k-cleaned_phase3"



# Server train
# python examples/scripts/sft_trainer.py --model_name "/home/koko/vicuna-7b/Vicuna 7b llama2" --dataset_name "/home/koko/dataset" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1


#NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 1  examples/scripts/sft_trainer.py --model_name "/home/koko/vicuna-7b/Vicuna 7b llama2" --dataset_name "/home/koko/dataset" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1