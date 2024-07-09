import json
import torch
import logging
import argparse

from mingpt.utils import set_seed
from mingpt.dataset import SortDataset
from mingpt.model import GPT as minGPT
from mingpt.trainer import Trainer
from torch.utils.data.dataloader import DataLoader

from sigmaifgpt.model import GPT as sigmaifGPT
from sigmaifgpt.utils import Logger


def train(args: argparse.Namespace) -> None:
    # Instantiate Logger
    Logger()
    logging.info("Process started")

    # Extract consol arguments
    config_filepath = args.config_filepath
    logging.info(f"Config filepath: {config_filepath}")

    # Load configuration
    with open(config_filepath, "r") as f:
        config = json.load(f)
    logging.info(f"Config loaded: {config}")

    # Set seed
    seed = config["seed"]
    set_seed(seed)
    logging.info(f"Seed set to: {seed}")

    # Instantiate datasets
    sequenceLength = config["dataset"]["sequenceLength"]
    digitsNumber = config["dataset"]["digitsNumber"]
    train_dataset = SortDataset("train", sequenceLength, digitsNumber)
    test_dataset = SortDataset("test", sequenceLength, digitsNumber)

    # Instantiate model
    model_architecture = config["model"]["architecture"]
    model_type = config["model"]["type"]
    logging.info(f"Model architecture set to: {model_architecture}")
    logging.info(f"Model type set to: {model_type}")
    if model_architecture == "minGPT":
        model_config = minGPT.get_default_config()
        model_config.model_type = model_type
        model_config.vocab_size = train_dataset.get_vocab_size()
        model_config.block_size = train_dataset.get_block_size()

        model = minGPT(model_config)
    elif model_architecture == "sigmaifGPT":
        model_config = sigmaifGPT.get_default_config()
        model_config.model_type = model_type
        model_config.vocab_size = train_dataset.get_vocab_size()
        model_config.block_size = train_dataset.get_block_size()
        model_config.K = config["model"]["parameters"]["K"]
        model_config.aggregation_threshold = config["model"]["parameters"]["aggregationThreshold"]
        
        model = sigmaifGPT(model_config)
    else:
        error_msg = f"Unknown '{model_architecture}' model architecture"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Instantiate trainer
    trainer_config = Trainer.get_default_config()
    trainer_config.learning_rate = 5e-4
    trainer_config.max_iters = 2000
    trainer_config.num_workers = 0
    
    trainer = Trainer(trainer_config, model, train_dataset)
    trainer.set_callback("on_batch_end", batch_end_callback)

    # Train model
    logging.info(f"Training started")
    trainer.run()
    logging.info(f"Training ended")

    # Evaluate model
    logging.info(f"Evaluation started")
    model.eval()
    with torch.no_grad():
        score_trn = eval_split(model, trainer, train_dataset, max_batches=50)
        score_tst = eval_split(model, trainer, test_dataset,  max_batches=50)
    logging.info(f"Evaluation ended")
    logging.info("Process ended")

def eval_split(model: torch.nn.Module, trainer: Trainer, dataset: SortDataset, max_batches: int) -> torch.Tensor:
    n = dataset.length
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # Isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # Let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False) # Using greedy argmax, not sampling
        sol_candidate = cat[:, n:] # Isolate the filled in sequence
        # Compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # Only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                logging.warning("Model claims that %s sorted is %s but ground truth is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    logging.info("%s score: %d/%d = %.2f%% correct" % (dataset.split.upper(), rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

def batch_end_callback(trainer: Trainer) -> None:
    if trainer.iter_num % 100 == 0:
        logging.info(f"iter_dt {trainer.iter_dt * 1000:.2f} ms; iter {trainer.iter_num}; train_loss {trainer.loss.item():.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath")
    args = parser.parse_args()
    train(args)
