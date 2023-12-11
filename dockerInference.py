import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, GenerationMixin, LlamaTokenizer
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training


class LLM:
    def __init__(self, early_stopping=True, bos_token_id=None, eos_token_id=None, pad_token_id=None, do_sample=True, top_k=30, top_p=0.95, temperature=0.5, max_new_tokens=512, repetition_penalty=1.2):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.bnb_config = None
        self.peft_model_id = None

        self.model = None
        torch.cuda.empty_cache()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        peft_model_id = "/app/final_output-44.17k-13b"

        self.model = LlamaForCausalLM.from_pretrained("/app/vicuna-13b-v1.5/", device_map="cuda:0",
                                                      quantization_config=bnb_config)
        self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = prepare_model_for_kbit_training(self.model)
        self.tokenizer = LlamaTokenizer.from_pretrained("/app/vicuna-13b-v1.5/")
        self.tokenizer.add_special_tokens(
            {'eos_token': "</s>", 'bos_token': '<s>'})

        self.PRE_PROMPT = '''You are a food ordering chatbot. Your name is Yum. You must follow the instruction section (###TASK) and execute it perfectly while using the system message section (###SYSTEM_MESSAGE) to help you with the answer. Be specific, fun, and do not add anything from what you don't see in the system message section (###SYSTEM_MESSAGE).'''

        self.gen_config = GenerationConfig()
        self.gen_config.early_stopping = early_stopping
        self.gen_config.bos_token_id = self.tokenizer.bos_token_id
        self.gen_config.eos_token_id = self.tokenizer.eos_token_id
        self.gen_config.pad_token_id = self.tokenizer.pad_token_id
        self.gen_config.do_sample = do_sample
        self.gen_config.top_k = top_k
        self.gen_config.top_p = top_p
        self.gen_config.temperature = temperature
        self.gen_config.max_new_tokens = max_new_tokens
        self.gen_config.repetition_penalty = repetition_penalty

    def predict(self, instruction, systemMessage):

        input_text = f'''<s>[INST]<<SYS>>{self.PRE_PROMPT}<</SYS>>
###TASK:\n
{instruction}
###SYSTEM_MESSAGE:\n
{systemMessage}[/INST]\n
###RESPONSE:\n'''

        inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs["input_ids"].to(
                "cuda:0"), generation_config=self.gen_config)
            output = self.tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            res = output[output.find("###RESPONSE:") +
                         len("###RESPONSE:"):]
            return res,input_text
