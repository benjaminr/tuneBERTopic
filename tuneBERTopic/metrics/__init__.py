from .supervised import precision, rouge, bleu
from .unsupervised import coherence, silhouette


evaluation_metrics = {
    "coherence": coherence,
    "precision": precision,
    "silhouette": silhouette,
    "rouge": rouge,
    "bleu": bleu
}