import sys
import os
import torch
import copy
import tqdm


def merge(model_parts):
    print("Merging Model")
    if len(model_parts) == 1:
        return model_parts[0]
    
    new_model = {}
    for k, v in model_parts[0].items():
        assert len(v.size()) < 3
        if len(v.shape) == 2 and "role_embeds.weight" not in k:
            if 'project.weight' in k:
                part = v.shape[0] // 3
                tmp_model = [[model[k][i*part:(i+1)*part, :] for model in model_parts] for i in range(3)]
                new_model[k] = torch.cat([x for y in tmp_model for x in y], dim=0)
            elif 'project_q.weight' in k:
                new_model[k] = torch.cat([model[k] for model in model_parts], dim=0)
            elif 'project_kv.weight' in k:
                part = v.shape[0] // 2
                tmp_model = [[model[k][i*part:(i+1)*part, :] for model in model_parts] for i in range(2)]
                new_model[k] = torch.cat([x for y in tmp_model for x in y], dim=0)
            elif any([x in k for x in ['word_embeds.weight', 'dense_relu_dense.wi_1.weight', 'dense_relu_dense.wi_0.weight', 'lm_head.weight']]):
                new_model[k] = torch.cat([model[k] for model in model_parts], dim=0)
            else:
                new_model[k] = torch.cat([model[k] for model in model_parts], dim=1)
        else:
            new_model[k] = v
    
    return new_model


def split(model, mp):
    print("Spliting model")
    if mp == 1:
        return [model]

    new_model_parts = []
    if mp == 1:
        new_model_parts.append(model)
        return new_model_parts

    for i in tqdm.tqdm(range(mp)):
        new_model = {}
        for k, v in model.items():
            assert len(v.shape) < 3
            if len(v.shape) == 2 and "role_embeds.weight" not in k:
                if 'project.weight' in k:
                    part = v.shape[0] // mp // 3
                    new_model[k] = torch.cat([v[i*part:(i+1)*part, :], v[(i+mp)*part:(i+1+mp)*part, :], v[(i+2*mp)*part:(i+1+2*mp)*part, :]], 0)
                elif 'project_q.weight' in k:
                    part = v.shape[0] // mp
                    new_model[k] = v[i*part:(i+1)*part, :]
                elif 'project_kv.weight' in k:
                    part = v.shape[0] // mp // 2
                    new_model[k] = torch.cat([v[i*part:(i+1)*part, :], v[(i+mp)*part:(i+1+mp)*part, :]], 0)
                elif any([x in k for x in ['word_embeds.weight', 'dense_relu_dense.wi_1.weight', 'dense_relu_dense.wi_0.weight', 'lm_head.weight']]):
                    part = v.shape[0] // mp
                    new_model[k] = v[i*part:(i+1)*part, :]
                else: # o.weight
                    part = v.shape[1] // mp
                    new_model[k] = v[:, i*part:(i+1)*part]
            else:
                new_model[k] = v
        
        new_model_parts.append(new_model)
    return new_model_parts


def main():

    preserve_keys = [
        "lr_scheduler",
        "skipped_steps",
        "global_steps",
        "global_samples",
        "dp_world_size",
        "iteration",
        "np_rng_state",
        "random_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
        
    ]

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    target_mp = int(sys.argv[3])
    
    with open(os.path.join(input_dir, "latest_checkpointed_iteration.txt")) as f:
        iter = int(f.read())
    
    model_dir = os.path.join(input_dir, str(iter))
    filenames = os.listdir(model_dir)
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[2]))
    filenames = [os.path.join(model_dir, x) for x in filenames]
    print("Model files:", filenames)

    ckpt_parts = [torch.load(filename, map_location="cpu") for filename in filenames]
    model_parts = [ckpt["module"] for ckpt in ckpt_parts]
    new_model = merge(model_parts)
    new_model_parts = split(new_model, target_mp)

    assert len(new_model_parts) == target_mp

    ckpt_new = {}
    for k, v in ckpt_parts[0].items():
        if k != 'module':
            if k in preserve_keys:
                ckpt_new[k] = copy.deepcopy(v)
            elif k == "mp_world_size":
                ckpt_new[k] = target_mp
            else:
                ckpt_new[k] = None
        ckpt_new['module'] = {}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, str(iter)), exist_ok=True)
    with open(os.path.join(output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(iter) + "\n")

    for i, model_part in enumerate(new_model_parts):
        ckpt_new['module'] = model_part
        torch.save(ckpt_new, os.path.join(output_dir, str(iter), "mp_rank_0{}_model_states.pt".format(i)))


if __name__ == "__main__":
    main()