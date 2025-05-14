from transformers import pipeline

# pipeline groups together three steps: preprocessing, passing the inputs through the model, and postprocessing
# pipeline choses model adopted to the specified task by default

# initialize the checkpoints (model weights) by pipeline
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

print(results)