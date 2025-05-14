# instantiate the checkpoint using the model architecture directly

from transformers import AutoTokenizer, AutoModelForMaskedLM

# AutoClasses are by design architecture-agnostic (it adopts best architecture ex. not only the bert ones)
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")