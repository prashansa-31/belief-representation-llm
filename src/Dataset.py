import pandas as pd
import json
import csv


class Dataset:
    def __init__(self):
        pass

    def load_csv_to_json(self, path):
        df = pd.read_csv(path)
        json_data = json.loads(df.to_json(orient="records", indent=2))
        return json_data

    def load_csv_data(self, path):
        csv_rows = None
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            csv_rows = list(reader)
        return csv_rows

    def update_possible_ans_list(self, prompt, possible_answers):
        updated_possible_answers = []
        prompt = prompt.split("\n")
        for line in prompt:
            if line.startswith("A.") or line.startswith("B."):
                if possible_answers[0] in line:
                    updated_possible_answers.append(line.strip())
                elif possible_answers[1] in line:
                    updated_possible_answers.append(line.strip())
        return updated_possible_answers

    def make_all_prompts(self, json_data, prompt_type, dataset_name):
        prompts_correctFB = []
        prompts_incorrectFB = []
        correct_false_belief = incorrect_false_belief = target_key = None
        possible_answers = []
        current_story_index = 0
        # to only include belief related prompts
        if dataset_name == "sally-anne":
            excluded_qa_type_list = ["reality", "memory"]
        else:
            excluded_qa_type_list = ["reality", "assumption"]
        for data in json_data:
            if dataset_name == "sally-anne":
                data["short_answer"] = data["short_answer"][:-1]
            if current_story_index != data["story_index"]:
                current_story_index = data["story_index"]
                correct_false_belief = incorrect_false_belief = target_key = None
                possible_answers_temp = []
            if data["question_type"] == excluded_qa_type_list[0]:
                possible_answers_temp.append(data["short_answer"])
            if data["question_type"] == excluded_qa_type_list[1]:
                possible_answers_temp.append(data["short_answer"])
            if data["question_type"] not in excluded_qa_type_list:
                if prompt_type in ["mc_prompt", "tf_prompt", "tfr_prompt"]:
                    possible_answers = self.update_possible_ans_list(
                        data[prompt_type], possible_answers_temp
                    )
                else:
                    possible_answers = possible_answers_temp
                if data["short_answer"] in possible_answers[0]:
                    correct_false_belief = possible_answers[0]
                    incorrect_false_belief = possible_answers[1]
                else:
                    incorrect_false_belief = possible_answers[0]
                    correct_false_belief = possible_answers[1]
                prompts_correctFB.append(
                    {
                        "story": data[prompt_type],
                        "belief": correct_false_belief,
                        "yl": 1,
                        "belief_type": data["question_type"],
                    }
                )
                prompts_incorrectFB.append(
                    {
                        "story": data[prompt_type],
                        "belief": incorrect_false_belief,
                        "yl": 0,
                        "belief_type": data["question_type"],
                    }
                )

        return prompts_correctFB, prompts_incorrectFB
