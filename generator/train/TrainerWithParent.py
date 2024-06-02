import torch
import torch.nn.functional as F
from transformers import Trainer
from adan import Adan

from .train_utils.lr_scheduler import BitnetLRScheduler

class TrainerWithParent(Trainer):
    
    def create_optimizer(self):
        self.optimizer = Adan(
            self.model.model.parameters(),
            lr=self.config_opt_sch.train.learning_rate,
            weight_decay=self.config_opt_sch.train.weight_decay,
            betas=(0.98, 0.92, 0.99),
            eps=self.config_opt_sch.train.adam_epsilon,
        )
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        self.lr_scheduler = BitnetLRScheduler(
            optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            second_lr=1e-3,
            second_weight_decay=0,
        )
        return self.lr_scheduler
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps, self.optimizer)
        
        
    def __init__(self, config_opt_sch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model_parent = model_parent.to("cuda")
        self.config_opt_sch = config_opt_sch

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Identical to HF transformers compute_loss, but with extra logging.
        """

        # outputs = model(**inputs, output_attentions=True, return_attentions_before_softmax=True)
        # with torch.no_grad():
        #     outputs_parent = self.model_parent(**inputs, output_attentions=True, return_attentions_before_softmax=True)
        calculate_two = True
        if calculate_two:
            outputs, outputs_parent, log_p_parent = model.train_forward(inputs, True, True)
            
            p_model = F.softmax(outputs["logits"], dim=-1)
            log_p_model = F.log_softmax(outputs["logits"], dim=-1)
            
            loss_last = -(p_model * (log_p_parent - log_p_model)).sum(axis=-1).mean()
            
            
            
            

            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # dict_loss = {}
            # if self.state.global_step % 50 == 0:
            #     dict_loss = {}
            #     dict_loss["general_loss"] = loss.item()
            #     dict_loss["loss_last"] = loss_last.item()
                # self.log({"general_loss": loss.item()})
                # self.log({"loss_last": -loss_last.item()})
            
            loss = 0.6 * loss + 0.4 * loss_last
            # mask = torch.ones(outputs_parent.attentions[0].shape).triu(diagonal=1).to(model.device)
            for i in range(len(outputs_parent.attentions)):
                # loss_heads = 0
                head_parent = outputs_parent.attentions[i]
                head_model = F.log_softmax(outputs.attentions[i][1], dim=-1)
                p_head_model = outputs.attentions[i][0]
                loss_heads = (p_head_model * (head_model - head_parent)).sum(axis=-1).mean()
                # for j in range(len(outputs_parent.attentions[i][0])):
                #     loss_head = 0
                #     head_parent = outputs_parent.attentions[i][:, j]
                #     # with torch.no_grad():
                #     #     head_parent = F.log_softmax(outputs_parent.attentions[i][:, j], dim=-1)
                #     head_model = F.log_softmax(outputs.attentions[i][1][:, j], dim=-1)
                #     p_head_model = outputs.attentions[i][0][:, j]
                #     for k in range(len(outputs_parent.attentions[i][0, j])):
                #         loss_head -= (
                #             p_head_model[:, k, :k+1] * (head_parent[:, k, :k+1] - head_model[:, k, :k+1])
                #         ).sum(axis=-1).mean()
                #     loss_heads += loss_head / len(outputs_parent.attentions[i][0, j])
                # loss_heads = loss_heads / len(outputs_parent.attentions[i][0])
                # if self.state.global_step % 50 == 0:
                    # dict_loss[f"loss_{i}_head"] = loss_heads.item()
                    # self.log({f"loss_{i}_head": -loss_heads.item()})
                loss += 0.1 * loss_heads
                
            # if self.state.global_step % 50 == 0:
                # dict_loss["loss"] = loss.item()
                # self.log(dict_loss)
            return (loss, outputs) if return_outputs else loss