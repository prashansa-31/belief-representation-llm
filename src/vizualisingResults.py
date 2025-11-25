from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy.stats import ttest_rel, f_oneway
import scipy.stats as st
import os
from .gdv import cmpGDV


class Visualize:
    def __init__(self):
        pass

    def plotProbing(
        self,
        plot,
        folder,
        perspective,
        id,
        iter,
        f1_score_all,
        val_acc_all,
        test_acc_mlp,
        f1_score_mlp,
        f1_score_Y_shuffled,
        val_acc_Y_shuffled,
        test_acc_mlp_Y_shuffled,
        f1_score_mlp_Y_shuffled,
    ):
        if plot == "heatmap":
            self.plot_heatmap(
                f1_score_all,
                "Linear Probe f1 score",
                save_path=f"{folder}/{perspective}_f1_score_{id}",
            )
            self.plot_heatmap(
                val_acc_all,
                "Linear Probe Acc.",
                save_path=f"{folder}/{perspective}_lr_acc_{id}",
            )
            # plot_heatmap(roc_auc_all,
            #             "ROC AUC Val",
            #             save_path= f"{folder}/{perspective}_val_auc_{id}" )
            self.plot_heatmap(
                test_acc_mlp,
                "Non-linear Probe acc",
                save_path=f"{folder}/{perspective}_mlp_acc_{id}",
            )
            self.plot_heatmap(
                f1_score_mlp,
                "Non-linear Probe f1 score",
                save_path=f"{folder}/{perspective}_f1_score_mlp_{id}",
            )

    def plot_heatmap(self, ht, name, save_path=None):
        # Increase global font size for all text elements
        plt.rcParams.update({"font.size": 22})

        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(15, 12))

        # Create a heatmap using seaborn
        sns.heatmap(
            ht,
            ax=ax,
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"drawedges": False, "shrink": 0.5},
            cbar=True,
            square=True,
            annot=True,
            annot_kws={"size": 6},
            fmt=".2f",
        )

        # Customize the colorbar
        cbar = ax.collections[0].colorbar
        cbar.outline.set_linewidth(2)  # Set colorbar outline width

        # Set the ticks for x and y axes with specified interval
        ax.set_xticks(np.arange(0.5, ht.shape[1], 5))
        ax.set_yticks(np.arange(0.5, ht.shape[0], 5))

        # Set the tick labels for x and y axes with specified interval and keep x-axis labels horizontal
        ax.set_xticklabels(np.arange(0, ht.shape[1], 5), rotation=0)
        ax.set_yticklabels(np.arange(0, ht.shape[0], 5))

        # Set axis labels and title with increased padding and font size
        ax.set_xlabel("Head", fontsize=24, labelpad=20)
        ax.set_ylabel("Layer", fontsize=24, labelpad=20)
        # ax.set_title(name, fontsize=28)

        # Reinstate axis lines with specified linewidth
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_visible(True)
            ax.spines[axis].set_linewidth(2)

        # Optionally save the figure as a PDF with vectorized content
        if save_path:
            # print(save_path + '.png')
            plt.savefig(save_path + ".png", format="png", bbox_inches="tight")

        # Clear the current figure's memory to prevent resource leaks
        plt.close(fig)

    def plot_linegraph(self, ht, pt_title, save_path=None):
        num_layers = len(ht)
        num_heads = len(ht[0])
        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(10, 7))

        for head in range(num_heads):
            head_acc = [ht[layer][head] for layer in range(num_layers)]
            ax.plot(range(num_layers), head_acc, label=f"Head {head}", marker="s")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(pt_title)
        plt.grid()
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        # ax.grid(True)
        # ax.tight_layout()
        # Optionally save the figure as a PDF with vectorized content
        if save_path:
            # print(save_path + '.png')
            plt.savefig(save_path + ".png", format="png", bbox_inches="tight")

        # Clear the current figure's memory to prevent resource leaks
        plt.close(fig)

    def plot_all_linegraph(
        self,
        train_data_list,
        test_data_list,
        train_data_list_control,
        test_data_list_control,
        pt_title,
        save_path=None,
    ):
        data_label_list = ["L", "NL"]
        ht = train_data_list[0]
        num_layers = len(ht)
        num_heads = len(ht[0])
        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(10, 8))
        # plot random guess acc at 0.5
        y = [0.5] * num_layers
        ax.plot(
            range(num_layers),
            y,
            linestyle="dashed",
            label=f"random",
            color="k",
            linewidth=1,
        )

        # for head in range(num_heads):
        #     head_acc = [ht[layer][head] for layer in range(num_layers)]
        #     ax.plot(range(num_layers), head_acc, label=f"Head {head}", marker = "s")
        data_color_list = ["tab:blue", "tab:orange", "tab:green"]
        temp_control_label_list = ["control-L", "control-NL"]
        temp_control_color = ["#000000", "#808080"]
        if True:
            # viz. train data
            save_path = f"{save_path}_train_test"
        for i in range(len(train_data_list)):
            if True:
                # viz. train data
                ax.plot(
                    range(num_layers),
                    train_data_list[i],
                    linestyle="dashed",
                    label=f"f1_{data_label_list[i]}",
                    color=data_color_list[i],
                )
            # viz. test data
            ax.plot(
                range(num_layers),
                test_data_list[i],
                marker="s",
                label=f"acc_{data_label_list[i]}",
                color=data_color_list[i],
            )
            ax.plot(
                range(num_layers),
                test_data_list_control[i],
                label=f"{temp_control_label_list[i]}",
                color=temp_control_color[i],
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        # ax.set_xlim(0, 15)
        ax.set_title(pt_title)
        plt.grid()
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        # ax.grid(True)
        # ax.tight_layout()
        # Optionally save the figure as a PDF with vectorized content
        if save_path:
            # print(save_path + '.png')
            plt.savefig(save_path + ".png", format="png", bbox_inches="tight")

        # Clear the current figure's memory to prevent resource leaks
        plt.close(fig)

    def dimentionalityReduction(self, proj_key, X_value, y_labels):
        return_proj = ""
        if proj_key == "PCA":
            x_pca_proj = PCA(n_components=2).fit_transform(X_value)
            intra, inter, gdv = cmpGDV(x_pca_proj, y_labels)
            return_proj = x_pca_proj
        elif proj_key == "TSNE":
            tsne = TSNE(
                n_components=2,
                max_iter=1000,
                perplexity=min(30, (X_value.shape[0] - 1) // 3),
            )
            x_tsne_proj = tsne.fit_transform(X_value)
            intra, inter, gdv = cmpGDV(x_tsne_proj, y_labels)
            return_proj = x_tsne_proj
        elif proj_key == "UMAP":
            umap_model = umap.UMAP(
                n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean"
            )
            x_umap_proj = umap_model.fit_transform(X_value)
            intra, inter, gdv = cmpGDV(x_umap_proj, y_labels)
            return_proj = x_umap_proj

        return return_proj, intra, inter, gdv

    def plot_individual_DR(
        self,
        proj_key,
        dataset_name,
        X_value,
        label_y,
        belief_types,
        plt_title,
        filename,
        save_path=None,
    ):
        x_projected_val, intra, inter, gdv = self.dimentionalityReduction(
            proj_key, X_value, label_y
        )
        y = []
        if proj_key == "PCA":
            x_label = "PC1"
            y_label = "PC2"
        elif proj_key == "TSNE":
            x_label = "t-SNE1"
            y_label = "t-SNE2"
        elif proj_key == "UMAP":
            x_label = "UMAP1"
            y_label = "UMAP2"
        for i in range(len(label_y)):
            if label_y[i] == 1:
                y.append("blue")
            else:
                y.append("red")
        legend_elements = [
            Patch(color="blue", label="correct belief"),
            Patch(color="red", label="incorrect belief"),
        ]
        plt.figure(figsize=(9, 7))
        plt.scatter(x_projected_val[:, 0], x_projected_val[:, 1], c=y, alpha=0.6)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        temp_plt_title = f"{plt_title} | GDV:{gdv:.3f}"
        plt.title(temp_plt_title, fontsize=14)
        filename = f"{save_path}/{filename}.png"
        plt.savefig(filename, format="png", bbox_inches="tight")
        plt.close()
        return intra, inter, gdv

    def plot_all_layers_DR(
        self,
        all_attn_head_mtx,
        Y_label,
        belief_types,
        id,
        dataset_name,
        probe_results_path,
        model_name_folder,
        prompt_type,
    ):
        num_layer = all_attn_head_mtx.shape[1]
        if id == "AH":
            num_attn_heads = all_attn_head_mtx.shape[2]
            plt_title = f"Layer <l>, Attention Head <ah>"
            filename = f"<vis>_Layer<l>_AH<ah>"
        else:
            num_attn_heads = 1
            plt_title = f"Layer <l>"
            filename = f"<vis>_Layer<l>"
        total_plots = num_layer * num_attn_heads
        vis_list = ["PCA", "TSNE", "UMAP"]
        separability_dict_pca = {
            "layers": list(range(num_layer)),
            "intra": [],
            "inter": [],
            "gdv": [],
        }
        separability_dict_tsne = {
            "layers": list(range(num_layer)),
            "intra": [],
            "inter": [],
            "gdv": [],
        }
        separability_dict_umap = {
            "layers": list(range(num_layer)),
            "intra": [],
            "inter": [],
            "gdv": [],
        }
        for vis in vis_list:
            save_path = f"{probe_results_path}/{dataset_name}/{model_name_folder}_{prompt_type}/{vis}_{id}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            print(f"{vis}...")
            for layer_idx in tqdm(range(num_layer)):  # each layer
                for attn_head_idx in range(num_attn_heads):  # each AH
                    if len(all_attn_head_mtx.shape) == 4:
                        X_plot = all_attn_head_mtx[:, layer_idx, attn_head_idx, :]
                    else:
                        X_plot = all_attn_head_mtx[:, layer_idx, :]
                    filename_updated = (
                        filename.replace("<vis>", vis)
                        .replace("<l>", str(layer_idx + 1))
                        .replace("<ah>", str(attn_head_idx + 1))
                    )
                    plt_title_updated = plt_title.replace(
                        "<l>", str(layer_idx + 1)
                    ).replace("<ah>", str(attn_head_idx + 1))
                    intra, inter, gdv = self.plot_individual_DR(
                        vis,
                        dataset_name,
                        X_plot,
                        Y_label,
                        belief_types,
                        plt_title_updated,
                        filename_updated,
                        save_path,
                    )
                    if vis == "PCA":
                        separability_dict_pca["gdv"].append(gdv)
                        separability_dict_pca["intra"].append(intra)
                        separability_dict_pca["inter"].append(inter)
                    elif vis == "TSNE":
                        separability_dict_tsne["gdv"].append(gdv)
                        separability_dict_tsne["intra"].append(intra)
                        separability_dict_tsne["inter"].append(inter)
                    else:
                        separability_dict_umap["gdv"].append(gdv)
                        separability_dict_umap["intra"].append(intra)
                        separability_dict_umap["inter"].append(inter)

        # print(separability_dict)
        return separability_dict_pca, separability_dict_tsne, separability_dict_umap

    def plot_acc_lineplot(
        self, filepath, filename, model_name, data_type, num_layers, layer_tm
    ):
        df_dataset = pd.read_csv(f"{filepath}/{filename}")
        prompt_color_list = [
            "tab:red",
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:cyan",
            "tab:pink",
        ]
        prompt_list = [
            "qa_prompt",
            "comp_prompt",
            "mc_prompt",
            "fb_prompt",
            "tf_prompt",
            "tfr_prompt",
        ]

        num_layers = num_layers
        fig_nl, ax_nl = plt.subplots(figsize=(10, 8))
        # fig_auc, ax_auc = plt.subplots(figsize=(10, 8))
        # non-linear
        temp_filter_df = []
        x_data_control_nl = []
        x_data_control_l = []
        y_data = list(range(1, num_layers + 1))
        for i in range(len(prompt_list)):
            temp_filter_df = df_dataset[
                (df_dataset["probe_type"] == "nonlinear-MLP")
                & (df_dataset["prompt_type"] == f"{prompt_list[i]}")
                & (df_dataset["layers_tm"] == f"{layer_tm}")
            ]
            x_data_acc_nl = temp_filter_df["accuracy"].to_list()
            x_data_f1_nl = temp_filter_df["f1_scores"].to_list()
            x_data_auc_nl = temp_filter_df["roc_auc"].to_list()
            x_data_control_nl.append(temp_filter_df["control_acc"].to_list())
            ax_nl.plot(
                y_data,
                x_data_acc_nl,
                label=f"{prompt_list[i]}",
                color=prompt_color_list[i],
            )
            ax_nl.plot(
                y_data,
                x_data_f1_nl,
                linestyle="dashed",
                color=prompt_color_list[i],
            )
            # ax_auc.plot(
            #     range(num_layers),
            #     x_data_auc_nl,
            #     label=f"{prompt_list[i]}",
            #     color=prompt_color_list[i],
            # )
        x_control_mean = np.mean(x_data_control_nl, axis=0)
        ax_nl.plot(
            y_data,
            x_control_mean,
            color="tab:gray",
            label="mean_control",
        )
        ax_nl.plot(
            y_data,
            [0.5] * num_layers,
            linestyle="dashed",
            color="tab:gray",
            label="random",
        )
        ax_nl.set_xlabel("Layer")
        ax_nl.set_ylabel("Accuracy")
        ax_nl.set_ylim(0.4, 1.0)
        ax_nl.set_title(f"{model_name}")
        plt.grid(True)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        plt.savefig(
            f"{filepath}/acc_plot/{model_name}_{data_type}_{layer_tm}_NL.png",
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig_nl)

        # linear
        fig_l, ax_l = plt.subplots(figsize=(10, 8))
        temp_filter_df = []
        for i in range(len(prompt_list)):
            temp_filter_df = df_dataset[
                (df_dataset["probe_type"] == "linear-LR")
                & (df_dataset["prompt_type"] == f"{prompt_list[i]}")
                & (df_dataset["layers_tm"] == f"{layer_tm}")
            ]
            x_data_acc_l = temp_filter_df["accuracy"].to_list()
            x_data_f1_l = temp_filter_df["f1_scores"].to_list()
            x_data_auc_l = temp_filter_df["roc_auc"].to_list()
            x_data_control_l.append(temp_filter_df["control_acc"].to_list())
            ax_l.plot(
                y_data,
                x_data_acc_l,
                label=f"{prompt_list[i]}",
                color=prompt_color_list[i],
            )
            ax_l.plot(
                y_data,
                x_data_f1_l,
                linestyle="dashed",
                color=prompt_color_list[i],
            )
            # ax_auc.plot(
            #     range(1, num_layers + 1),
            #     x_data_auc_l,
            #     linestyle="dashed",
            #     color=prompt_color_list[i],
            # )

        x_control_mean_l = np.mean(x_data_control_l, axis=0)
        ax_l.plot(
            y_data,
            x_control_mean_l,
            color="tab:gray",
            label="mean_control",
        )
        ax_l.plot(
            y_data,
            [0.5] * num_layers,
            linestyle="dashed",
            color="tab:gray",
            label="random",
        )
        ax_l.set_xlabel("Layer")
        ax_l.set_ylabel("Accuracy")
        ax_l.set_ylim(0.4, 1.0)
        ax_l.set_title(f"{model_name}")
        plt.grid(True)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        plt.savefig(
            f"{filepath}/acc_plot/{model_name}_{data_type}_{layer_tm}_L.png",
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig_l)

    def plot_gdv_lineplot(self, filepath, filename, data_type, num_layers, layer_tm):
        df_dataset = pd.read_csv(f"{filepath}/{filename}")
        prompt_color_list = [
            "tab:red",
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:cyan",
            "tab:pink",
        ]
        prompt_list = [
            "qa_prompt",
            "comp_prompt",
            "mc_prompt",
            "fb_prompt",
            "tf_prompt",
            "tfr_prompt",
        ]
        column_list = ["pca_gdv", "tsne_gdv", "umap_gdv"]

        num_layers = num_layers
        # non-linear
        temp_filter_df = []
        y_data = list(range(1, num_layers + 1))
        for col in column_list:
            fig_l, ax_l = plt.subplots(figsize=(10, 8))
            temp_filter_df = []
            for i in range(len(prompt_list)):
                temp_filter_df = df_dataset[
                    (df_dataset["prompt_type"] == f"{prompt_list[i]}")
                    & (df_dataset["layers_tm"] == f"{layer_tm}")
                ]
                gdv_list = temp_filter_df[col].to_list()
                ax_l.plot(
                    y_data,
                    gdv_list,
                    label=f"{prompt_list[i]}",
                    color=prompt_color_list[i],
                )
            ax_l.set_xlabel("Layer")
            ax_l.set_ylabel("gdv")
            ax_l.set_ylim(-0.5, 0.5)
            ax_l.set_title(f"{col}")
            plt.grid(True)
            plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
            plt.savefig(
                f"{filepath}/gdv_plot/{data_type}_{layer_tm}_{col}.png",
                format="png",
                bbox_inches="tight",
            )
            plt.close(fig_l)
