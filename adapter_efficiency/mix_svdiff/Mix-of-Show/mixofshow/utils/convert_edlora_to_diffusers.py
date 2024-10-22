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

def merge_SVDiff_into_weight(original_state_dict, lora_state_dict, model_type, alpha):
    def get_spectral_C_name(original_layer_name):
        if model_type == 'text_encoder':
            spectral_C_name = original_layer_name.replace('q_proj.weight', 'q_proj.spectral_C') \
                .replace('k_proj.weight', 'k_proj.spectral_C') \
                .replace('v_proj.weight', 'v_proj.spectral_C') \
                .replace('out_proj.weight', 'out_proj.spectral_C') \
                .replace('fc1.weight', 'fc1.spectral_C') \
                .replace('fc2.weight', 'fc2.spectral_C')
        else:
            spectral_C_name = k.replace('to_q.weight', 'to_q.spectral_C') \
                .replace('to_k.weight', 'to_k.spectral_C') \
                .replace('to_v.weight', 'to_v.spectral_C') \
                .replace('to_out.0.weight', 'to_out.0.spectral_C') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.spectral_C') \
                .replace('ff.net.2.weight', 'ff.net.2.spectral_C') \
                .replace('proj_out.weight', 'proj_out.spectral_C') \
                .replace('proj_in.weight', 'proj_in.spectral_C')

        return spectral_C_name

    assert model_type in ['unet', 'text_encoder']
    new_state_dict = copy.deepcopy(original_state_dict)
    load_cnt = 0
    for k in new_state_dict.keys():
        spectral_C_name = get_spectral_C_name(k)
        U_name = spectral_C_name.replace('spectral_C', 'U')
        S_name = spectral_C_name.replace('spectral_C', 'S')
        V_name = spectral_C_name.replace('spectral_C', 'V')
        if spectral_C_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            spectral_C_params = lora_state_dict[spectral_C_name].to(original_params.device)
            U_params = lora_state_dict[U_name].to(original_params.device)
            S_params = lora_state_dict[S_name].to(original_params.device)
            V_params = lora_state_dict[V_name].to(original_params.device)
            pad_S = S_params + alpha*spectral_C_params
            if len(original_params.shape) == 4:
                raise Exception('')
            else:
                spectral_param = U_params@pad_S.diag()@V_params.T
            new_state_dict[k] = spectral_param
    print(f'load {load_cnt} Spectrals of {model_type}')
    return new_state_dict




def convert_edlora(pipe, state_dict, enable_edlora, alpha=0.6):

    state_dict = state_dict['params'] if 'params' in state_dict.keys() else state_dict

    # step 1: load embedding
    if 'new_concept_embedding' in state_dict and len(state_dict['new_concept_embedding']) != 0:
        pipe, new_concept_cfg = load_new_concept(pipe, state_dict['new_concept_embedding'], enable_edlora)

    # step 2: merge lora weight to unet
    unet_lora_state_dict = state_dict['unet']
    pretrained_unet_state_dict = pipe.unet.state_dict()
    updated_unet_state_dict = merge_SVDiff_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet',  alpha=alpha)
    pipe.unet.load_state_dict(updated_unet_state_dict)

    # step 3: merge lora weight to text_encoder
    text_encoder_lora_state_dict = state_dict['text_encoder']
    pretrained_text_encoder_state_dict = pipe.text_encoder.state_dict()
    updated_text_encoder_state_dict = merge_SVDiff_into_weight(pretrained_text_encoder_state_dict, text_encoder_lora_state_dict, model_type='text_encoder', alpha=alpha)
    pipe.text_encoder.load_state_dict(updated_text_encoder_state_dict)

    return pipe, new_concept_cfg
