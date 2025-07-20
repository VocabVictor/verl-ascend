# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import inspect
from dataclasses import asdict
from typing import Dict

from verl.utils import get_logger

logger = get_logger()


class TrainerFactory:
    TRAINER_MAPPING = {
        'causal_lm': 'verl.trainers.Seq2SeqTrainer',
        'seq_cls': 'verl.trainers.Trainer',
        'embedding': 'verl.trainers.EmbeddingTrainer',
        'reranker': 'verl.trainers.RerankerTrainer',
        'generative_reranker': 'verl.trainers.RerankerTrainer',
        'dpo': 'verl.trainers.DPOTrainer',
        'orpo': 'verl.trainers.ORPOTrainer',
        'kto': 'verl.trainers.KTOTrainer',
        'cpo': 'verl.trainers.CPOTrainer',
        'rm': 'verl.trainers.RewardTrainer',
        'ppo': 'verl.trainers.PPOTrainer',
        'grpo': 'verl.trainers.GRPOTrainer',
        'gkd': 'verl.trainers.GKDTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        'causal_lm': 'verl.trainers.Seq2SeqTrainingArguments',
        'seq_cls': 'verl.trainers.TrainingArguments',
        'embedding': 'verl.trainers.TrainingArguments',
        'reranker': 'verl.trainers.TrainingArguments',
        'generative_reranker': 'verl.trainers.TrainingArguments',
        'dpo': 'verl.trainers.DPOConfig',
        'orpo': 'verl.trainers.ORPOConfig',
        'kto': 'verl.trainers.KTOConfig',
        'cpo': 'verl.trainers.CPOConfig',
        'rm': 'verl.trainers.RewardConfig',
        'ppo': 'verl.trainers.PPOConfig',
        'grpo': 'verl.trainers.GRPOConfig',
        'gkd': 'verl.trainers.GKDConfig',
    }

    @staticmethod
    def get_cls(args, mapping: Dict[str, str]):
        if hasattr(args, 'rlhf_type'):
            train_method = args.rlhf_type
        else:
            train_method = args.task_type
        module_path, class_name = mapping[train_method].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def get_trainer_cls(cls, args):
        return cls.get_cls(args, cls.TRAINER_MAPPING)

    @classmethod
    def get_training_args(cls, args):
        training_args_cls = cls.get_cls(args, cls.TRAINING_ARGS_MAPPING)
        args_dict = asdict(args)
        parameters = inspect.signature(training_args_cls).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        args._prepare_training_args(args_dict)
        return training_args_cls(**args_dict)
