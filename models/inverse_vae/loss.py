import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=1.0, gamma=1.0):
        """
        Custom loss function for the inverse problem in polymer design.

        Args:
            alpha (float): Weight for material similarity adjustment.
            beta (float): Weight for proportion loss.
            gamma (float): Weight for temperature profile loss.
        """
        super(InverseLoss, self).__init__()
        self.alpha = alpha  # Weight for material similarity
        self.beta = beta    # Weight for proportion loss
        self.gamma = gamma  # Weight for temperature profile loss

    def forward(self, pred, target, material_similarity_scores):
        """
        Compute the loss for the inverse problem.

        Args:
            pred (dict): Dictionary of predicted values with keys 'materials', 'proportions', 'temperature_profile'.
            target (dict): Dictionary of target values with keys 'material_ids', 'proportions', 'temp_profile'.
            material_similarity_scores (torch.Tensor): Tensor of material similarity scores.

        Returns:
            torch.Tensor: Total loss.
        """
        # Ensure similarity scores are in the range [0, 1]
        material_similarity_scores = torch.clamp(material_similarity_scores, min=0, max=1)

        # Material cross-entropy with similarity adjustment
        material_loss = F.cross_entropy(pred['materials'], target['material_ids'], reduction='none')
        material_loss *= (1 + self.alpha * material_similarity_scores).squeeze()
        material_loss = material_loss.mean()  # Average over batch

        # Proportions MSE
        proportion_loss = F.mse_loss(pred['proportions'], target['proportions'])

        # Temperature profile loss (if using LSTM)
        temp_loss = F.mse_loss(pred['temperature_profile'], target['temp_profile'])

        # Combine losses with weights
        total_loss = material_loss + self.beta * proportion_loss + self.gamma * temp_loss

        return total_loss