import copy
import torch


def load_new_concept(pipe, new_concept_embedding, enable_edlora=True):
    new_concept_cfg = {}

    for idx, (concept_name, concept_embedding) in enumerate(new_concept_embedding.items()):
        if enable_edlora:
            num_new_embedding = 16
        else:
            num_new_embedding = 1
        new_token_names = [f'<new{idx * num_new_embedding + layer_id}>' for layer_id in range(num_new_embedding)]
        num_added_tokens = pipe.tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == len(new_token_names), 'some token is already in tokenizer'
        new_token_ids = [pipe.tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

        # init embedding
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
        token_embeds[new_token_ids] = concept_embedding.clone().to(token_embeds.device, dtype=token_embeds.dtype)
        print(f'load embedding: {concept_name}')

        new_concept_cfg.update({
            concept_name: {
                'concept_token_ids': new_token_ids,
                'concept_token_names': new_token_names
            }
        })

    return pipe, new_concept_cfg

def merge_oft_into_weight(original_state_dict, lora_state_dict, model_type, block_share, oft_r,alpha):
    def cayley(data):
        r, c = list(data.shape)
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def cayley_batch(data):
        b, r, c = data.shape
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q
    def get_R_name(original_layer_name):
        if model_type == 'text_encoder':
            R_name = original_layer_name.replace('q_proj.weight', 'q_proj.R') \
                .replace('k_proj.weight', 'k_proj.R') \
                .replace('v_proj.weight', 'v_proj.R') \
                .replace('out_proj.weight', 'out_proj.R') \
                .replace('fc1.weight', 'fc1.R') \
                .replace('fc2.weight', 'fc2.R')
        else:
            R_name = k.replace('to_q.weight', 'to_q.R') \
                .replace('to_k.weight', 'to_k.R') \
                .replace('to_v.weight', 'to_v.R') \
                .replace('to_out.0.weight', 'to_out.0.R') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.R') \
                .replace('ff.net.2.weight', 'ff.net.2.R') \
                .replace('proj_out.weight', 'proj_out.R') \
                .replace('proj_in.weight', 'proj_in.R')

        return R_name
    def block_diagonal(R):
        if block_share:
            blocks = [R] * oft_r
        else:
            blocks = [R[i, ...] for i in range(oft_r)]
        A = torch.block_diag(*blocks)
        return A

    assert model_type in ['unet', 'text_encoder']
    new_state_dict = copy.deepcopy(original_state_dict)
    load_cnt = 0
    for k in new_state_dict.keys():
        R_name = get_R_name(k)
        if R_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            R_params = lora_state_dict[R_name].to(original_params.device)
            dtype = R_params.dtype
            if block_share:
                orth_rotate = cayley(R_params)
            else:
                orth_rotate = cayley_batch(R_params)
            block_diagonal_matrix = block_diagonal(orth_rotate)
            fix_filt = original_params
            fix_filt = torch.transpose(fix_filt, 0, 1)
            filt = torch.mm(alpha*block_diagonal_matrix, fix_filt.to(dtype))
            filt = torch.transpose(filt, 0, 1)
            if len(original_params.shape) == 4:
                raise Exception('')
            else:
                new_state_dict[k] = filt
    print(f'load {load_cnt} Spectrals of {model_type}')
    return new_state_dict


def convert_edlora(pipe, state_dict, enable_edlora, alpha=0.6, oft_r=64):

    state_dict = state_dict['params'] if 'params' in state_dict.keys() else state_dict

    # step 1: load embedding
    if 'new_concept_embedding' in state_dict and len(state_dict['new_concept_embedding']) != 0:
        pipe, new_concept_cfg = load_new_concept(pipe, state_dict['new_concept_embedding'], enable_edlora)

    # step 2: merge lora weight to unet
    unet_lora_state_dict = state_dict['unet']
    pretrained_unet_state_dict = pipe.unet.state_dict()
    updated_unet_state_dict = merge_oft_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet',  block_share=True,oft_r=oft_r, alpha=alpha)
    pipe.unet.load_state_dict(updated_unet_state_dict)

    # step 3: merge lora weight to text_encoder
    text_encoder_lora_state_dict = state_dict['text_encoder']
    pretrained_text_encoder_state_dict = pipe.text_encoder.state_dict()
    updated_text_encoder_state_dict = merge_oft_into_weight(pretrained_text_encoder_state_dict, text_encoder_lora_state_dict, model_type='text_encoder', block_share=True, oft_r=oft_r,alpha=alpha)
    pipe.text_encoder.load_state_dict(updated_text_encoder_state_dict)

    return pipe, new_concept_cfg
