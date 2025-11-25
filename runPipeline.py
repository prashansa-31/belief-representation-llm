import numpy as np
from pathlib import Path
import pandas as pd
from src.Dataset import Dataset
from src.hfLM import EvalLLM
from src.vizualisingResults import Visualize
from src.probing import Probing
from datetime import datetime
import yaml
import os


def updateCSVDict(
    acc_list,
    f1_scores_list,
    roc_auc_list,
    probe_type,
    layer_tm,
    prompt_type,
    model_csv_dict,
    acc_ci_list,
    control_acc_list,
):
    for i in range(len(acc_list)):
        model_csv_dict["layers_number"].append(i + 1)
        model_csv_dict["layers_tm"].append(layer_tm)
        model_csv_dict["prompt_type"].append(prompt_type)
        model_csv_dict["probe_type"].append(probe_type)
        model_csv_dict["accuracy"].append(f"{acc_list[i][0]:.3f}")
        # model_csv_dict["acc_ci"].append(
        #     f"{acc_list[i][0]:.3f} ± {acc_ci_list[i][0]:.3f}"
        # )
        model_csv_dict["f1_scores"].append(f"{f1_scores_list[i][0]:.3f}")
        model_csv_dict["roc_auc"].append(f"{roc_auc_list[i][0]:.3f}")
        model_csv_dict["control_acc"].append(f"{control_acc_list[i][0]:.3f}")

    return model_csv_dict


def updateSepDict(
    layer_tm,
    prompt_type,
    model_csv_dict,
    pca_gdv,
    pca_inter,
    pca_intra,
    tsne_gdv,
    tsne_inter,
    tsne_intra,
    umap_gdv,
    umap_inter,
    umap_intra,
):
    for i in range(len(pca_gdv)):
        model_csv_dict["layers_number"].append(i + 1)
        model_csv_dict["layers_tm"].append(layer_tm)
        model_csv_dict["prompt_type"].append(prompt_type)
        model_csv_dict["pca_gdv"].append(f"{pca_gdv[i]:.3f}")
        model_csv_dict["pca_inter"].append(f"{pca_inter[i]:.3f}")
        model_csv_dict["pca_intra"].append(f"{pca_intra[i]:.3f}")
        model_csv_dict["tsne_gdv"].append(f"{tsne_gdv[i]:.3f}")
        model_csv_dict["tsne_inter"].append(f"{tsne_inter[i]:.3f}")
        model_csv_dict["tsne_intra"].append(f"{tsne_intra[i]:.3f}")
        model_csv_dict["umap_gdv"].append(f"{umap_gdv[i]:.3f}")
        model_csv_dict["umap_inter"].append(f"{umap_inter[i]:.3f}")
        model_csv_dict["umap_intra"].append(f"{umap_intra[i]:.3f}")

    return model_csv_dict


def pipelineCUDA(
    model,
    model_layers_list,
    prompt_type_list,
    model_path,
    sally_anne_json,
    dataset_name,
    attention_output_path,
    no_iter=1,
):
    evalLLM_obj = EvalLLM()
    dataset_obj = Dataset()
    print(f"All Representations Extracted for {model_path} model")
    for prompt_type in prompt_type_list:
        for iter in range(no_iter):
            prompts_correctFB, prompts_incorrectFB = dataset_obj.make_all_prompts(
                sally_anne_json, prompt_type, dataset_name
            )
            print(
                "Correct Belief prompt:"
                + prompt_type
                + " :\n"
                + str(prompts_correctFB[0])
            )
            print(
                "In-Correct Belief prompt:"
                + prompt_type
                + " :\n"
                + str(prompts_incorrectFB[0])
            )
            evalLLM_obj.extract_representations(
                model,
                dataset_name,
                prompts_correctFB,
                prompts_incorrectFB,
                prompt_type,
                attention_output_path=attention_output_path,
            )


def pipelineCPU(
    model_layers_list,
    prompt_type_list,
    model_path,
    dataset_name,
    model_name_folder,
    attention_output_path,
    probe_results_path,
    model_accuracy_csv_filepath,
    model_acc_dict,
    model_sep_dict,
    no_iter=1,
):
    probing_obj = Probing()
    viz_obj = Visualize()
    for layer in model_layers_list:
        print(f"All Representations Extracted for {model_path} model at {layer} layer:")
        for prompt_type in prompt_type_list:
            for iter in range(no_iter):
                file_data = np.load(
                    attention_output_path
                    / dataset_name
                    / f"{dataset_name}_probe_training_data_{prompt_type}_{layer}.npy",
                    allow_pickle=True,
                ).item()  # load saved probe training data
                ip_prompts = file_data["all_input_prompts"]
                # print(f"Total {len(ip_prompts)} data loaded.")
                data = file_data["all_layer_activations"]  # X-data
                train_labels = file_data["all_training_labels"]  # Y-data
                belief_types = file_data["all_belief_types"]
                print(f"Length of dataset: {str(len(data))}")
                folder = f"{probe_results_path}/{dataset_name}/{model_name_folder}_{prompt_type}/"
                (Path(folder)).mkdir(parents=True, exist_ok=True)
                if layer == "AH":
                    plot = "heatmap"
                    controltask_flag = False
                else:
                    plot = "linegraph"
                    controltask_flag = (
                        True  # make it true when want to plot control taks
                    )
                print(f"{prompt_type} {layer} Probing...")
                (
                    lr_ci,
                    lr_f1_score,
                    lr_val_acc,
                    lr_roc_auc,
                    lr_control_acc,
                    mlp_ci,
                    mlp_val_acc,
                    mlp_f1_score,
                    mlp_roc_auc,
                    mlp_control_acc,
                ) = probing_obj.train_probe(
                    plot,
                    data,
                    train_labels,
                    belief_types,
                    "llm",
                    folder,
                    dataset_name=dataset_name,
                    id=layer,
                    iter=iter + 1,
                )
                data_matrix = np.stack(data)
                model_acc_dict = updateCSVDict(
                    lr_val_acc,
                    lr_f1_score,
                    lr_roc_auc,
                    "linear-LR",
                    layer,
                    prompt_type,
                    model_acc_dict,
                    lr_ci,
                    lr_control_acc,
                )
                model_acc_dict = updateCSVDict(
                    mlp_val_acc,
                    mlp_f1_score,
                    mlp_roc_auc,
                    "nonlinear-MLP",
                    layer,
                    prompt_type,
                    model_acc_dict,
                    mlp_ci,
                    mlp_control_acc,
                )
                print(f"{prompt_type} {layer} Visualizing...")
                (
                    separability_dict_pca,
                    separability_dict_tsne,
                    separability_dict_umap,
                ) = viz_obj.plot_all_layers_DR(
                    data_matrix,
                    train_labels,
                    belief_types,
                    layer,
                    dataset_name,
                    probe_results_path,
                    model_name_folder,
                    prompt_type,
                )
                pca_gdv = list(separability_dict_pca["gdv"])
                pca_inter = list(separability_dict_pca["inter"])
                pca_intra = list(separability_dict_pca["intra"])
                tsne_gdv = list(separability_dict_tsne["gdv"])
                tsne_inter = list(separability_dict_tsne["inter"])
                tsne_intra = list(separability_dict_tsne["intra"])
                umap_gdv = list(separability_dict_umap["gdv"])
                umap_inter = list(separability_dict_umap["inter"])
                umap_intra = list(separability_dict_umap["intra"])
                model_sep_dict = updateSepDict(
                    layer,
                    prompt_type,
                    model_sep_dict,
                    pca_gdv,
                    pca_inter,
                    pca_intra,
                    tsne_gdv,
                    tsne_inter,
                    tsne_intra,
                    umap_gdv,
                    umap_inter,
                    umap_intra,
                )
                print("done.")

    final_model_acc_dict = pd.DataFrame(model_acc_dict)
    final_model_sep_dict = pd.DataFrame.from_dict(model_sep_dict)
    return final_model_acc_dict, final_model_sep_dict


def initModelValues():
    with open("./config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    hf_token = config["extraction"]["TOKEN"]
    model_path = config["extraction"]["MODEL_ID"]
    model_name_folder = "llama3.2_1b"
    ROOT_PATH, attention_output_path, probe_results_path = initPath(config)
    return (
        ROOT_PATH,
        attention_output_path,
        probe_results_path,
        model_path,
        hf_token,
        model_name_folder,
    )


def initPath(config):
    ROOT_PATH = config["extraction"]["REPO_PATH"]
    # Initialise output paths
    attention_output_path = Path(f"{ROOT_PATH}/output/probe_training_data")
    attention_output_path.mkdir(parents=True, exist_ok=True)
    (attention_output_path / "smarties").mkdir(parents=False, exist_ok=True)
    (attention_output_path / "sally-anne").mkdir(parents=False, exist_ok=True)

    probe_results_path = Path(f"{ROOT_PATH}/output/probe_results")
    probe_results_path.mkdir(parents=True, exist_ok=True)
    (probe_results_path / "smarties").mkdir(parents=True, exist_ok=True)
    (probe_results_path / "sally-anne").mkdir(parents=True, exist_ok=True)

    return ROOT_PATH, attention_output_path, probe_results_path


def triggerPipeline(splitRunFlag=False):
    model_acc_filename = "model_accuracy.csv"
    model_sep_filename = "model_sep.csv"
    (
        ROOT_PATH,
        attention_output_path,
        probe_results_path,
        model_path,
        hf_token,
        model_name_folder,
    ) = initModelValues()

    evalLLM_obj = EvalLLM()
    # selecting dataset and other variables
    (
        model_path,
        temperature,
        hf_token,
        prompt_type_list,
        prompt_type,
        model_name_folder,
    ) = evalLLM_obj.init_model_paras(model_path, hf_token, model_name_folder)

    dataset_list = ["sally-anne/Sally-Anne_prompt.csv", "smarties/smarties_prompt.csv"]
    prompt_type_list = [
        "qa_prompt",
        "comp_prompt",
        "mc_prompt",
        "fb_prompt",
        "tf_prompt",
        "tfr_prompt",
    ]
    model_layers_list = ["SAL", "MLP", "HS"]
    device = "cpu"
    # model = evalLLM_obj.load_model(model_path, device, temperature, hf_token)
    for current_runingData in dataset_list:
        print(f"Currently running Data: {current_runingData}")
        print(f"Currently running for model: {model_path}")
        csv_path = f"{ROOT_PATH}/data/{current_runingData}"
        dataset_obj = Dataset()
        csv_json = dataset_obj.load_csv_to_json(csv_path)
        dataset_name = current_runingData.split("/")[0]

        # show_model_details()  # upadate function
        device = "cpu"
        temp_filename = model_acc_filename.split(".")
        # addTimestamp = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        model_accuracy_csv_filepath = Path(
            f"{probe_results_path}/{dataset_name}/{temp_filename[0]}_{model_name_folder}_{dataset_name}.{temp_filename[1]}"
        )
        model_acc_dict = pd.read_csv(f"{ROOT_PATH}/{model_acc_filename}").to_dict(
            orient="list"
        )
        model_sep_dict = pd.read_csv(f"{ROOT_PATH}/{model_sep_filename}").to_dict(
            orient="list"
        )

        # pipelineCUDA(
        #     model,
        #     model_layers_list,
        #     prompt_type_list,
        #     model_path,
        #     csv_json,
        #     dataset_name,
        #     attention_output_path,
        # )
        final_model_acc_dict, final_model_sep_dict = pipelineCPU(
            model_layers_list,
            prompt_type_list,
            model_path,
            dataset_name,
            model_name_folder,
            attention_output_path,
            probe_results_path,
            model_accuracy_csv_filepath,
            model_acc_dict,
            model_sep_dict,
        )
        model_acc_path = Path(f"{ROOT_PATH}/output/model_accuracy_{dataset_name}.csv")
        final_model_acc_dict.to_csv(model_acc_path)
        print(f"Model Accuracy saved in file-{model_acc_path}")
        model_sep_path = Path(
            f"{ROOT_PATH}/output/model_separability_{dataset_name}.csv"
        )
        final_model_sep_dict.to_csv(model_sep_path)
        print(f"Model Separability saved in file-{model_acc_path}")

    test_create_imag(ROOT_PATH)


def test_create_imag(
    root_path,
):
    viz_obj = Visualize()
    filepath = f"{root_path}/output"
    filename_list = os.listdir(filepath)
    layer_tm_list = ["SAL", "MLP", "HS"]
    for layer_tm in layer_tm_list:
        for filename in filename_list:
            temp = filename.split("_")
            num_layers = 16
            model_name = "llama3.2_1b"
            if "model_accuracy" in filename and filename.endswith(".csv"):
                data_type = temp[2].split(".")[0]
                viz_obj.plot_acc_lineplot(
                    filepath, filename, model_name, data_type, num_layers, layer_tm
                )
            elif "model_separability" in filename and filename.endswith(".csv"):
                data_type = temp[2].split(".")[0]
                viz_obj.plot_gdv_lineplot(
                    filepath, filename, data_type, num_layers, layer_tm
                )


if __name__ == "__main__":
    triggerPipeline()
