# Copyright (c) 2023, Sejeong Lee. All rights reserved.
from argparse import ArgumentParser
import functools
import pickle

import pytorch_lightning as pl
import torch
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
from argoverse.evaluation.competition_util import generate_forecasting_h5
import evalai_api as evalai

if __name__ == "__main__":
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pickle_prediction", type=bool, default=True)
    # submission arguments
    parser.add_argument("--h5_filename", type=str, default="argoverse_forecasting_test")
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--method_description", type=str)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, parallel=True
    )
    test_dataset = ArgoverseV1Dataset(
        root=args.root, split="test", local_radius=model.hparams.local_radius
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    predictions = functools.reduce(
        lambda former_dict, latter_dict: {**former_dict, **latter_dict},
        trainer.predict(model, dataloader),
    )

    if args.pickle_prediction:
        filename = "predictions.pkl"
        print(f"Pickling predictions to {filename}")
        pickle.dump(predictions, open(filename, "wb"))

    output_path = "competition_files/"
    generate_forecasting_h5(predictions, output_path, filename=args.h5_filename)

    submission_detail = evalai.SubmissionDetails(
        method_name=args.method_name,
        method_description=args.method_description,
        project_url="",
        publication_url="",
    )
    evalai.submit(
        phase_id=941,
        challenge_id=454,
        file=open(f"./competition_files/{args.h5_filename}.h5", "rb"),
        large=True,
        public=False,
        private=True,
        submission_details=submission_detail,
        annotation=False,
    )
