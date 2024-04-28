# -*- coding: utf-8 -*-

#!pip install octis
#!pip install FuzzyTM

import argparse
import os
import json
import time
import Path

import numpy as np

from octis.dataset.dataset import Dataset

from octis.models.LDA import LDA
from octis.models.ETM import ETM
from octis.models.CTM import CTM
from octis.models.LSI import LSI
from octis.models.NMF import NMF
from octis.models.ProdLDA import ProdLDA
from octis.models.NeuralLDA import NeuralLDA

from FuzzyTM import FLSA_W
from FuzzyTM import FLSA

from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.optimization.optimizer import Optimizer

from octis.models.model import AbstractModel


class BaseModel(AbstractModel):
    def __init__(
            self,
            model_name,
            input_file,
            num_topics = 20,
            num_words = 20,
            word_weighting = 'normal',
            cluster_method = 'fcm',
            svd_factors = 2
            ):
        super().__init__()
        self.model_name = model_name
        self.hyperparameters = dict()
        self.hyperparameters["num_topics"] = num_topics
        self.hyperparameters["num_words"] = num_words
        self.hyperparameters["word_weighting"] = word_weighting
        self.hyperparameters["cluster_method"] = cluster_method
        self.hyperparameters["svd_factors"] = int(svd_factors)

    def train_model(
            self,
            dataset,
            hyperparameters=None,
            top_words=20
            ):

        if hyperparameters is None:
            hyperparameters = self.hyperparameters.copy()

        if "num_topics" not in hyperparameters:
            hyperparameters["num_topics"] = self.hyperparameters["num_topics"]

        hyperparameters["input_file"] = dataset.get_corpus()

        if isinstance(hyperparameters['svd_factors'], np.integer):
            hyperparameters['svd_factors'] = int(hyperparameters['svd_factors'])

        print(f"Before model init, svd_factors type: {type(hyperparameters['svd_factors'])}, value: {hyperparameters['svd_factors']}")

        if self.model_name == 'FLSA':
            self.trained_model = FLSA(**hyperparameters)
        elif self.model_name == 'FLSA_W':
            self.trained_model = FLSA_W(**hyperparameters)

        result = {}

        result["topic-word-matrix"],result["topic-document-matrix"] = self.trained_model.get_matrices()
        result["topics"] = self.trained_model.show_topics(representation = 'words')
        return result

class OptimizeAndTrainModels:
    def __init__(
            self, dataset_idx, model_name, optimization_runs = 75, model_runs=5
            ):

        self.optimization_runs = optimization_runs
        self.model_runs = model_runs
        self.model_name = model_name
        self.optimization_results = None

        self.datasets = {
            1: 'BBC_News',
            2: '20NewsGroup',
            3: 'DBLP',
            4: 'M10'
            }

        if dataset_idx not in self.datasets:
            raise ValueError("Choosea a valid dataset_idx")

        self.dataset_idx = dataset_idx
        self.dataset_name = self.datasets[self.dataset_idx]

        self.models = {
            'LDA': {
                'model': LDA,
                'search_space': {
                    "decay": Real(0.5,1),
                    "alpha": Categorical(['asymmetric','auto','symmetric']),
                    "gamma_threshold": Categorical([0.0001,0.001,0.01])
                    }
                },
            'ETM': {
                'model': ETM,
                'search_space': {
                    "dropout": Real(0,0.95),
                    "num_layers": Integer(1,3),
                    "num_neurons": Categorical([100, 200, 300])},
                },
            'CTM': {
                'model': CTM,
                'search_space': {
                    "dropout": Real(0,0.95),
                    "num_layers": Integer(1,3),
                    "num_neurons": Categorical([100, 200, 300])},
                },
            'LSI': {
                'model' : LSI,
                'search_space' : {
                    'extra_samples': Categorical([100, 200, 300]),
                    'decay': Real(0.5,1.5),
                    'power_iters': Integer(1,3)},
                },
            'NMF': {
                'model': NMF,
                'search_space' : {
                    "w_stop_condition": Categorical([0.00001, 0.0001, 0.001]),
                    "kappa": Real(0.5,1.5),
                    "h_stop_condition": Categorical([0.0001, 0.001,0.01])}
                },
            'NeuralLDA' : {
                'model' : NeuralLDA,
                'search_space' : {
                    "num_layers": Integer(1,3),
                    "num_neurons": Categorical([100, 200, 300]),
                    "dropout": Real(0.0, 0.95)}
                },
            "ProdLDA": {
                'model' : ProdLDA,
                'search_space' : {
                    "num_layers": Integer(1,3),
                    "num_neurons": Categorical([100, 200, 300]),
                    "dropout": Real(0.0, 0.95)}
                },
            'FLSA': {
                'model' : FLSA,
                'search_space' : {
                    "svd_factors" : Integer(2,5),
                    "word_weighting" : Categorical({"normal","idf","probidf","entropy"}),
                    "cluster_method" : Categorical({"fcm", "gk"})
                    }
                },
            'FLSA_W': {
                'model' : FLSA_W,
                'search_space' : {
                    "svd_factors" : Integer(2,5),
                    "word_weighting" : Categorical({"normal","idf","probidf","entropy"}),
                    "cluster_method" : Categorical({"fcm", "gk"})
                    }
                },
            }

        if model_name not in self.models:
            raise ValueError("Chood a valid method name.")

        octis_models = {
            'LDA': True, 'ETM': True, 'CTM': True, 'LSI': True, 'NMF': True, 'NeuralLDA': True, 'ProdLDA': True,
            'FLSA': False, 'FLSA_W': False
        }

        if model_name not in octis_models:
            raise ValueError('Invalid model_name')
        self.octis_model = octis_models[model_name]

        self.model_catalog = self.models[model_name]


    def optimize_model(self):
        dataset = Dataset()
        dataset.fetch_dataset(self.dataset_name)

        model_catalog = self.models[self.model_name]

        if self.octis_model:
            model = model_catalog['model'](
                num_topics=20,
                input_file = dataset.get_corpus())
        else:

            model = BaseModel(
                model_name = self.model_name,
                num_topics=20,
                input_file = dataset.get_corpus()
                )

        search_space = self.model_catalog['search_space']
        coherence = Coherence(texts=dataset.get_corpus(), measure = 'c_v')
        optimizer=Optimizer()

        start = time.time()
        self.optimization_results = optimizer.optimize(
            model,
            dataset,
            coherence,
            search_space,
            number_of_call=self.optimization_runs,
            model_runs=self.model_runs,
            save_models=True,
            extra_metrics=None,
            )
        end = time.time()
        duration = end - start
        print('Optimizing model took: ' + str(round(duration)) + ' seconds.')
        return self.optimization_results


    def train_topics(self):

        all_topics = {
            'model' : self.model_name,
            'dataset' : self.dataset_name,
        }

        if not self.optimization_results:
            self.optimization_results = self.optimize_model()

        hyper_parameter_values = self.optimization_results.x_iters

        max_coherence = max(self.optimization_results.func_vals)
        best_hyperparameter_idx = self.optimization_results.func_vals.index(max_coherence)

        optimized_hyperparameters = self._extract_optimized_hyperparameters(hyper_parameter_values,best_hyperparameter_idx)

        dataset = Dataset()
        dataset.fetch_dataset(self.dataset_name)

        if not self.octis_model:
            data = dataset.get_corpus()

        topics_library = {}
        for num_topics in range(10,101,10):

            iteration = {}

            if self.octis_model:

                for iteration in range(10):

                    model = self.model_catalog['model'](
                        num_topics = num_topics, **optimized_hyperparameters
                        )
                    trained_model = model.train_model(dataset)
                    topics = trained_model['topics']
                    iteration[num_topics] = topics

            else:

                for iteration in range(10):
                    model = self.model_catalog(
                        input_file = data,
                        num_topics = num_topics,
                        **optimized_hyperparameters
                    )
                    _,_ = model.get_matrices()
                    topics = model.show_topics()
                    iteration[num_topics] = topics

            topics_library[num_topics] = iteration

        all_topics['topics'] = topics_library

        return all_topics


    def _extract_optimized_hyperparameters( self, data_dict, index ):
        result_dict = {}

        for key, value_list in data_dict.items():
            try:
                result_dict[key] = value_list[index]
            except IndexError:
                print(f"Index {index} is out of range for key '{key}'.")
                return None

        return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optimize and train models based on provided parameters.")

    parser.add_argument("dataset_index", type=int, help="Index of the dataset to be used")
    parser.add_argument("model_name", type=str, help="Name of the model to be used")
    parser.add_argument("--optimization_runs", type=int, default=3, help="Number of optimization runs (default: 3)")
    parser.add_argument("--model_runs", type=int, default=1, help="Number of model training runs (default: 1)")
    parser.add_argument("--save_path", type=str, help="Path to save the topics output", default=None)

    args = parser.parse_args()

    dataset_index = args.dataset_index
    model_name = args.model_name
    optimization_runs = args.optimization_runs
    model_runs = args.model_runs
    save_path = args.save_path

    try:
        test = OptimizeAndTrainModels(dataset_index, model_name, optimization_runs, model_runs)
        topics = test.train_topics()

        if save_path:
            test.save_topics(topics, save_path)
        print(topics)
    except Exception as e:
        print(f"An error occurred: {e}")

    def save_topics(self, topics, file_path):

        os.makedirs(Path(file_path).parent, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(topics, f, indent=4)

        print(f"Topics saved to {file_path}")