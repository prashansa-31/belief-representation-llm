import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
)
from tqdm import tqdm
from src.vizualisingResults import Visualize


class Probing:
    def __init__(self):
        self.seed = 42
        self.controlTask_flag = True
        self.all_X = None
        self.all_y = None
        self.viz_obj = Visualize()
        self.linear_probe = None
        self.non_linear_probe = None

    def init_probe(self):
        self.linear_probe = LogisticRegression(
            random_state=self.seed, max_iter=1000, C=1, penalty="l2"
        )
        self.non_linear_probe = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=self.seed,
            ),
        )

    def confidence_interval(self, mean_list, confidence=0.95):
        """
        Compute mean and confidence interval for a list of scores.
        """
        import numpy as np
        from scipy import stats

        mean = np.mean(mean_list)
        std = np.std(mean_list)
        n = len(mean_list)
        h = stats.t.ppf((1 + confidence) / 2.0, n - 1) * std / np.sqrt(n)
        return h

    def train_probe(
        self,
        plot,
        attention_data,
        train_labels,
        belief_types,
        perspective,
        folder,
        dataset_name="default",
        id=None,
        iter=1,
    ):
        all_X, all_y = np.array(attention_data), np.array(train_labels)
        self.init_probe()

        (
            lr_ci,
            lr_f1_score,
            lr_val_acc,
            lr_roc_auc,
            mlp_ci,
            mlp_f1_score,
            mlp_val_acc,
            mlp_roc_auc,
        ) = self.probeAll(all_X, all_y)

        Y_shuffled = np.random.permutation(all_y)  # Random permutation test
        (
            lr_ci_y_shuff,
            lr_f1_score_y_shuff,
            lr_val_acc_y_shuff,
            lr_roc_auc_y_shuff,
            mlp_ci_y_shuff,
            mlp_f1_score_y_shuff,
            mlp_val_acc_y_shuff,
            mlp_roc_auc_y_shuff,
        ) = self.probeAll(all_X, Y_shuffled)

        self.viz_obj.plotProbing(
            plot,
            folder,
            perspective,
            id,
            iter,
            lr_f1_score,
            lr_val_acc,
            mlp_val_acc,
            mlp_f1_score,
            lr_f1_score_y_shuff,
            lr_val_acc_y_shuff,
            mlp_val_acc_y_shuff,
            mlp_f1_score_y_shuff,
        )
        return (
            lr_ci,
            lr_f1_score,
            lr_val_acc,
            lr_roc_auc,
            lr_val_acc_y_shuff,
            mlp_ci,
            mlp_val_acc,
            mlp_f1_score,
            mlp_roc_auc,
            mlp_val_acc_y_shuff,
        )

    def probeAll(self, all_X, all_y, train_size=0.8, test_size=0.2, seed=42):
        data_ids = np.arange(len(all_X))
        all_X_train, all_X_val, y_train, y_val, ids_train, ids_test = train_test_split(
            all_X,
            all_y,
            data_ids,
            train_size=train_size,
            test_size=test_size,
            stratify=all_y,
            random_state=seed,
        )
        # print(all_X_train.shape)
        if len(all_X_train.shape[1:]) > 2:
            num_layers, num_heads, head_dims = all_X_train.shape[1:]
        else:
            num_layers, head_dims = all_X_train.shape[1:]
            num_heads = 1
        lr_val_acc = np.zeros([num_layers, num_heads])
        lr_f1_score = np.zeros([num_layers, num_heads])
        lr_roc_auc = np.zeros([num_layers, num_heads])
        lr_ci = np.zeros([num_layers, num_heads])
        lr_p_value = np.zeros([num_layers, num_heads])
        mlp_val_acc = np.zeros([num_layers, num_heads])
        mlp_f1_score = np.zeros([num_layers, num_heads])
        mlp_roc_auc = np.zeros([num_layers, num_heads])
        mlp_ci = np.zeros([num_layers, num_heads])
        mlp_p_value = np.zeros([num_layers, num_heads])

        for layer in tqdm(range(num_layers)):
            for head in range(num_heads):
                # print(layer, head)
                if len(all_X_train.shape[1:]) > 2:
                    X_train = all_X_train[:, layer, head, :]
                    X_val = all_X_val[:, layer, head, :]
                    X_all = all_X[:, layer, head, :]
                else:
                    X_train = all_X_train[:, layer, :]
                    X_val = all_X_val[:, layer, :]
                    X_all = all_X[:, layer, :]

                (
                    lr_ci[layer][head],
                    lr_f1_score[layer][head],
                    lr_val_acc[layer][head],
                    lr_roc_auc[layer][head],
                    # lr_p_value[layer][head],
                ) = self.probe_single_layer_LR(X_all, all_y, seed)
                (
                    mlp_ci[layer][head],
                    mlp_f1_score[layer][head],
                    mlp_val_acc[layer][head],
                    mlp_roc_auc[layer][head],
                    # mlp_p_value[layer][head],
                ) = self.probe_single_layer_MLP(X_all, all_y)

        return (
            lr_ci,
            lr_f1_score,
            lr_val_acc,
            lr_roc_auc,
            mlp_ci,
            mlp_f1_score,
            mlp_val_acc,
            mlp_roc_auc,
        )

    def probe_single_layer_LR(self, all_X, all_y, seed=42, verbose=False):
        # Probe activations using Logistic regression
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
        scores = cross_validate(
            self.linear_probe, all_X, all_y, cv=cv, scoring=scoring, n_jobs=-1
        )
        val_acc = scores["test_accuracy"].mean()
        train_acc = scores["test_f1"].mean()
        roc_auc = scores["test_roc_auc"].mean()
        # Mean and 95% confidence interval
        ci = self.confidence_interval(scores["test_accuracy"])
        # print("Per-fold acc:", scores["test_accuracy"])
        return ci, train_acc, val_acc, roc_auc

    def probe_single_layer_MLP(self, all_X, all_y, verbose=False):
        # Probe activations using SVM-RBF
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
        scores = cross_validate(
            self.non_linear_probe, all_X, all_y, cv=cv, scoring=scoring, n_jobs=-1
        )
        val_acc = scores["test_accuracy"].mean()
        acc_train = scores["test_f1"].mean()
        roc_auc = scores["test_roc_auc"].mean()
        # Mean and 95% confidence interval
        ci = self.confidence_interval(scores["test_accuracy"])

        return ci, acc_train, val_acc, roc_auc
