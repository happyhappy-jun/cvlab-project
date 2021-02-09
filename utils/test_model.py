import os

import torch

from utils.utils import get_logger, is_logging_process

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

def test_model(cfg, model, test_loader, writer):
    logger = get_logger(cfg, os.path.basename(__file__))
    model.net.eval()
    total_test_loss = 0
    test_loop_len = 0
    total_test_accuracy = 0
    total_test = 0
    with torch.no_grad():
        for model_input, target in test_loader:
            model.feed_data(input=model_input, GT=target)
            output = model.run_network()
            loss_v = model.loss_f(output, model.GT)
            _, predicted = torch.max(output.data, 1)
            total_v = torch.tensor(int(target.size(0))).to('cuda')
            
            accuracy_v = torch.tensor(float((predicted == target.to('cuda')).sum().item())).to('cuda')
            

            # print(f"loss_v: {type(loss_v)}")
            # print(f"predicted: {type(accuracy_v)}")
            # print(f"target: {target.device}")
            # print(f"accuracy_v: {accuracy_v.device}")

            
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                torch.distributed.all_reduce(loss_v)
                loss_v /= torch.tensor(float(cfg.dist.gpus))

                # Aggregate accuracy_v from all GPUs. loss_v is set as the sum of all GPUs' accuracy_v.
                torch.distributed.all_reduce(accuracy_v)
                torch.distributed.all_reduce(total_v)

            total_test += total_v.to("cpu").item()
            total_test_loss += loss_v.to("cpu").item()
            total_test_accuracy += accuracy_v.to("cpu").item()

            test_loop_len += 1
        # print(f"total_v = {total_test}")
        # print(f"accuracy_v = {total_test_accuracy}")
        total_test_loss /= test_loop_len
        total_test_accuracy /= total_test

        if writer is not None:
            writer.logging_with_step(total_test_accuracy, model.step, "test_accuracy")
            writer.logging_with_step(total_test_loss, model.step, "test_loss")
            writer.logging_with_epoch(total_test_accuracy, model.step, model.epoch, "total_test_accuracy_per_epoch")
            writer.logging_with_epoch(total_test_loss, model.step, model.epoch, "test_loss_per_epoch")
        if is_logging_process():
            logger.info("Test Loss %.04f at step %d" % (total_test_loss, model.step))
