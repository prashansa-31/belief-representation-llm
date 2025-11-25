from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from transformers.generation.utils import GenerationConfig
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


class EvalLLM:
    def __init__(self):
        pass

    def init_model_paras(self, model_path, hf_token, model_name_folder):
        # Initialise LLM model parameters
        temperature = 0
        model_name_folder = model_name_folder
        prompt_type_list = [
            "qa_prompt",
            "comp_prompt",
            "mc_prompt",
            "fb_prompt",
            "tf_prompt",
            "tfr_prompt",
        ]
        prompt_type = prompt_type_list[2]
        model_path = model_path
        hf_token = hf_token
        return (
            model_path,
            temperature,
            hf_token,
            prompt_type_list,
            prompt_type,
            model_name_folder,
        )

    def show_model_details(self, model):
        cfg = getattr(model, "cfg", getattr(model.model, "config", None))

        # Extract info safely with fallback values
        model_info = {
            "Name": getattr(
                cfg, "model_name", getattr(cfg, "_name_or_path", "Unknown")
            ),
            "Number of layers": getattr(
                cfg, "n_layers", getattr(cfg, "num_hidden_layers", "Unknown")
            ),
            "Attention head": getattr(
                cfg, "n_heads", getattr(cfg, "num_attention_heads", "Unknown")
            ),
            "Hidden layer size": getattr(
                cfg, "d_model", getattr(cfg, "hidden_size", "Unknown")
            ),
            "Number of different tokens": getattr(
                cfg, "d_vocab", getattr(cfg, "vocab_size", "Unknown")
            ),
            "Context Size": getattr(
                cfg, "n_ctx", getattr(cfg, "max_position_embeddings", "Unknown")
            ),
        }

        print("=== Model Info ===")
        for k, v in model_info.items():
            print(f"{k}: {v}")

    def load_model(self, model_path, device, temperature, hf_token):
        if hf_token == "":
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, token=hf_token, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, token=hf_token, trust_remote_code=True
            )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            base_model.resize_token_embeddings(len(tokenizer))
        base_model.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = None
            base_model.generation_config.top_p = None
            base_model.generation_config.top_k = None
        else:
            base_model.generation_config.temperature = temperature
        base_model.to(device).eval()
        model = LanguageModel(base_model, tokenizer=tokenizer)
        self.show_model_details(model)
        return model

    def get_sublayers_activations(self, model, prompt):
        n_heads = model.model.config.num_attention_heads
        all_attention_heads = []
        all_hidden_states = []
        all_mlp_outputs = []
        all_attention_states = []
        with model.trace(prompt) as tracer:
            for layer in model.model.layers:
                all_attention_heads.append(layer.self_attn.output[0].save())
                all_hidden_states.append(
                    layer.output[0][:, -1, :].save()
                )  # extract hidden state of last token
                all_mlp_outputs.append(
                    layer.mlp.output[0].save()
                )  # extract activations from FFN of last token
                all_attention_states.append(
                    layer.self_attn.output[0][:, -1, :].save()
                )  # extract self attention of last token
        all_attention_heads_numpy = []
        all_hidden_states_numpy = []
        all_mlp_outputs_numpy = []
        all_mlp_zero_output = []
        all_mlp_mean_output = []
        all_mlp_std_output = []
        all_mlp_random_output = []
        all_attention_states_numpy = []
        # save the extracted activations
        for AS, HS, MLP, SAL in zip(
            all_attention_heads,
            all_hidden_states,
            all_mlp_outputs,
            all_attention_states,
        ):
            atts = AS.value[0].cpu().detach().numpy()
            ah_reshape = atts.reshape(atts.shape[0], n_heads, -1)
            all_attention_heads_numpy.append(ah_reshape)

            all_hidden_states_numpy.append(HS.value.cpu().detach().numpy())

            all_mlp_zero_output.append(torch.zeros_like(MLP))
            all_mlp_random_output.append(torch.randn_like(MLP))
            all_mlp_outputs_numpy.append(MLP.value[0].cpu().detach().numpy())
            mean_val = MLP.mean(dim=(0, 1), keepdim=True)
            std_val = MLP.std(dim=(0, 1), keepdim=True)
            all_mlp_mean_output.append(mean_val)
            all_mlp_std_output.append(std_val)

            atts = SAL.value.cpu().detach().numpy()
            all_attention_states_numpy.append(atts)

        all_attention_heads_numpy = np.array(all_attention_heads_numpy)
        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_mlp_outputs_numpy = np.array(all_mlp_outputs_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)

        return (
            all_attention_heads_numpy,
            all_hidden_states_numpy,
            all_mlp_outputs_numpy,
            all_attention_states_numpy,
        )

    def extract_representations(
        self,
        model,
        dataset_name,
        prompts_correctFB,
        prompts_incorrectFB,
        prompt_type,
        attention_output_path=Path("./output"),
    ):
        all_input_prompts_correct_belief = []
        all_input_prompts_incorrect_belief = []
        all_training_labels_correct_belief = []
        all_training_labels_incorrect_belief = []
        all_question_types_correct_belief = []
        all_question_types_incorrect_belief = []
        all_input_prompts = []
        all_training_labels = []
        all_layer_actiavtions_mhsa = []
        all_layer_activations_sal = []
        all_layer_actiavtions_ff = []
        all_layer_actiavtions_hs = []
        all_belief_types = []

        # print(f"Belief rep. for {prompt_type}")
        for idx in tqdm(range(len(prompts_correctFB))):
            # get correct false beleif attention states
            prompt = prompts_correctFB[idx]
            story = prompt["story"]
            belief = prompt["belief"]
            input_prompt = f"Story:{story}\nBelief:{belief}"
            all_input_prompts_correct_belief.append(input_prompt)
            all_training_labels_correct_belief.append(prompt["yl"])
            all_question_types_correct_belief.append(prompt["belief_type"])
            a_state, h_state, mlp_state, sa_layer = self.get_sublayers_activations(
                model, input_prompt
            )
            a_state = a_state[:, -1]
            all_layer_actiavtions_mhsa.append(a_state)
            all_layer_actiavtions_hs.append(h_state)
            all_layer_actiavtions_ff.append(mlp_state)
            all_layer_activations_sal.append(sa_layer)

            all_input_prompts.append(all_input_prompts_correct_belief[idx])
            all_training_labels.append(all_training_labels_correct_belief[idx])
            all_belief_types.append(all_question_types_correct_belief[idx])

            # get in-correct false beleif attention states
            prompt = prompts_incorrectFB[idx]
            story = prompt["story"]
            belief = prompt["belief"]
            input_prompt = f"Story:{story}\nBelief:{belief}"
            all_input_prompts_incorrect_belief.append(input_prompt)
            all_training_labels_incorrect_belief.append(prompt["yl"])
            all_question_types_incorrect_belief.append(prompt["belief_type"])
            a_state, h_state, mlp_state, sa_layer = self.get_sublayers_activations(
                model, input_prompt
            )
            a_state = a_state[:, -1]
            all_layer_actiavtions_mhsa.append(a_state)
            all_layer_actiavtions_hs.append(h_state)
            all_layer_actiavtions_ff.append(mlp_state)
            all_layer_activations_sal.append(sa_layer)
            all_input_prompts.append(all_input_prompts_incorrect_belief[idx])
            all_training_labels.append(all_training_labels_incorrect_belief[idx])
            all_belief_types.append(all_question_types_incorrect_belief[idx])

        probe_training_data = {
            "all_input_prompts": all_input_prompts,
            "all_training_labels": all_training_labels,
            "all_layer_activations": all_layer_actiavtions_mhsa,
            "all_belief_types": all_belief_types,
        }

        # Save each sublayer activation individually in their respective .np file to be processed later
        dataset_len = len(probe_training_data["all_input_prompts"])
        filename = f"{attention_output_path}/{dataset_name}/{dataset_name}_probe_training_data_{prompt_type}_AH.npy"
        filepath = Path(filename)
        np.save(filepath, probe_training_data, allow_pickle=True)

        probe_training_data = {
            "all_input_prompts": all_input_prompts,
            "all_training_labels": all_training_labels,
            "all_layer_activations": all_layer_activations_sal,
            "all_belief_types": all_belief_types,
        }

        dataset_len = len(probe_training_data["all_input_prompts"])
        filename = f"{attention_output_path}/{dataset_name}/{dataset_name}_probe_training_data_{prompt_type}_SAL.npy"
        filepath = Path(filename)
        np.save(filepath, probe_training_data, allow_pickle=True)

        probe_training_data = {
            "all_input_prompts": all_input_prompts,
            "all_training_labels": all_training_labels,
            "all_layer_activations": all_layer_actiavtions_ff,
            "all_belief_types": all_belief_types,
        }

        dataset_len = len(probe_training_data["all_input_prompts"])
        filename = f"{attention_output_path}/{dataset_name}/{dataset_name}_probe_training_data_{prompt_type}_MLP.npy"
        filepath = Path(filename)
        np.save(filepath, probe_training_data, allow_pickle=True)

        probe_training_data = {
            "all_input_prompts": all_input_prompts,
            "all_training_labels": all_training_labels,
            "all_layer_activations": all_layer_actiavtions_hs,
            "all_belief_types": all_belief_types,
        }

        dataset_len = len(probe_training_data["all_input_prompts"])
        filename = f"{attention_output_path}/{dataset_name}/{dataset_name}_probe_training_data_{prompt_type}_HS.npy"
        filepath = Path(filename)
        np.save(filepath, probe_training_data, allow_pickle=True)

        print("All Extracted activations Saved.")
        return probe_training_data, filename
