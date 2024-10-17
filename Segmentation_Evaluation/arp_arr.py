import torch
import torch.nn.functional as F


def precision_recall(segmentation_gt: torch.Tensor, segmentation_pred: torch.Tensor, mode: str, adjusted: bool):
    """ Compute the (Adjusted) Rand Precision/Recall.
    Args:
    segmentation_gt: Int tensor with shape (batch_size, height, width) containing the
    ground-truth segmentations.
    segmentation_pred: Int tensor with shape (batch_size, height, width) containing the
    predicted segmentations.
    mode: Either "precision" or "recall" depending on which metric shall be computed.
    adjusted: Return values for adjusted or non-adjusted metric.
    Returns:
    Float tensor with shape (batch_size), containing the (Adjusted) Rand
    Precision/Recall per sample.
    """
    max_classes = max(segmentation_gt.max(), segmentation_pred.max()) + 1
    oh_segmentation_gt = F.one_hot(segmentation_gt, max_classes)
    oh_segmentation_pred = F.one_hot(segmentation_pred, max_classes)

    coincidence = torch.einsum("bhwk,bhwc->bkc", oh_segmentation_gt, oh_segmentation_pred)
    coincidence_gt = coincidence.sum(-1)
    coincidence_pred = coincidence.sum(-2)

    m_squared = torch.sum(coincidence**2, (1, 2))
    m = torch.sum(coincidence, (1, 2))
    # How many pairs of pixels have the smae label assigned in ground-truth segmentation.
    P = torch.sum(coincidence_gt * (coincidence_gt - 1), -1)
    # How many pairs of pixels have the smae label assigned in predicted segmentation.
    Q = torch.sum(coincidence_pred * (coincidence_pred - 1), -1)

    expected_m_squared = (P + m) * (Q + m) / (m * (m - 2)) + (m**2 - Q - P -2 * m) / (m - 1)

    if mode == "precision":
        gamma = P + m
    elif mode == "recall":
        gamma = Q + m
    elif mode == 'ari':
        gamma = (P + Q + 2*m) / 2
    else:
        raise ValueError("Invalid mode.")
    if adjusted:
        return (m_squared - expected_m_squared) / (gamma - expected_m_squared)
    else:
        return (m_squared - m) / (gamma - m)