import torch
import os

from tqdm.asyncio import trange
from tqdm import tqdm
from transformers import AdamW

from data import TOKENIZER_PATH
from data_preprocessing.conversation_builder import get_conversations
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

from data_preprocessing.dataset import get_data_loader
from loggers.get_logger import get_logger

device = "cpu"
logger = get_logger("Training")


def main(model_name: str = "sberbank-ai/rugpt3small_based_on_gpt2",
         weight_decay: int = 0,
         lr: float = 5e-5,
         adam_epsilon: float = 1e-8,
         gradient_accumulation_steps: int = 8,
         n_epochs: int = 2,
         warmup_steps: int = 0,
         max_norm: int = 1,
         save_every: int = 5,
         run_name: str = 'run1'):

    all_conversations = get_conversations()
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

    data_loader = get_data_loader(all_conversations, tokenizer, batch_size=64)
    model.to(device)
    tokenizer.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
         },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    t_total = len(data_loader) // gradient_accumulation_steps * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    epochs_trained = global_step // (len(data_loader) // gradient_accumulation_steps)
    steps_trained_in_current_epoch = global_step % (len(data_loader) // gradient_accumulation_steps)
    logger.info("Continuing training from checkpoint, will skip to saved global_step")
    logger.info(f"Continuing training from epoch {epochs_trained}")
    logger.info(f"Continuing training from global step {global_step}")
    logger.info(f"Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")

    # Training loop
    model.zero_grad()
    epoch_pbar = trange(epochs_trained, int(n_epochs))
    av_loss = 0
    for current_epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch [{current_epoch+1}/{n_epochs}]")
        pbar = tqdm(data_loader, position=0)
        for step, batch in enumerate(pbar):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            # the language model targets (labels) are the same as the input!
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss, *_ = model(inputs, labels=labels)
            loss.backward()
            tr_loss = loss.item()
            # Compute a running average of the loss
            av_loss = (step*av_loss + tr_loss)/(step + 1)
            pbar.set_description(f"Average loss: {av_loss:.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % save_every == 0 and global_step > 0:
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join('runs', run_name, "{}-{}".format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saving optimizer and scheduler states to {output_dir}")
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    # save model
    output_dir = os.path.join('runs', run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Saving model checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
