import random

# model1 = ["/dataset/f1d6ea5b/gyx-eva/eva2/results/public-8g-medium-0.01-0.01-1/", "/dataset/f1d6ea5b/gyx-eva/eva-origin/src/configs/model/eva_model_config_medium_attn_scale.json"]
model1 = ["/dataset/f1d6ea5b/gyx-eva/eva2/results/fake-medium-0.01-0.01-1/", "/dataset/f1d6ea5b/gyx-eva/eva-origin/src/configs/model/eva_model_config_medium_attn_scale.json"]
model2 = ["/dataset/f1d6ea5b/gyx-eva/eva2/results/9-28-medium-0.01-0.01-1-0.01", "/dataset/f1d6ea5b/gyx-eva/eva-origin/src/configs/model/eva_model_config_medium.json"]


seed = random.randint(0, 1000)

if random.random() < 0.5:
    with open("test1", "w") as f:
        f.write(model1[0] + " " + str(seed) + " " + model1[1])
    with open("test2", "w") as f:
        f.write(model2[0] + " " + str(seed) + " " + model2[1])
else:
    with open("test2", "w") as f:
        f.write(model1[0] + " " + str(seed) + " " + model1[1])
    with open("test1", "w") as f:
        f.write(model2[0] + " " + str(seed) + " " + model2[1])