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

def merge_spectralpart_into_weight(original_state_dict, lora_state_dict, model_type, alpha, top=True, idx=1):
    def get_spectral_A_name(original_layer_name):
        if model_type == 'text_encoder':
            spectral_A_name = original_layer_name.replace('q_proj.weight', 'q_proj.spectral_A') \
                .replace('k_proj.weight', 'k_proj.spectral_A') \
                .replace('v_proj.weight', 'v_proj.spectral_A') \
                .replace('out_proj.weight', 'out_proj.spectral_A') \
                .replace('fc1.weight', 'fc1.spectral_A') \
                .replace('fc2.weight', 'fc2.spectral_A')
        else:
            spectral_A_name = k.replace('to_q.weight', 'to_q.spectral_A') \
                .replace('to_k.weight', 'to_k.spectral_A') \
                .replace('to_v.weight', 'to_v.spectral_A') \
                .replace('to_out.0.weight', 'to_out.0.spectral_A') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.spectral_A') \
                .replace('ff.net.2.weight', 'ff.net.2.spectral_A') \
                .replace('proj_out.weight', 'proj_out.spectral_A') \
                .replace('proj_in.weight', 'proj_in.spectral_A')

        return spectral_A_name
    
    assert model_type in ['unet', 'text_encoder']
    new_state_dict = copy.deepcopy(original_state_dict)
    load_cnt = 0
    for k in new_state_dict.keys():
        spectral_A_name = get_spectral_A_name(k)
        spectral_B_name = spectral_A_name.replace('spectral_A', 'spectral_B')
        spectral_C_name = spectral_A_name.replace('spectral_A', 'spectral_C')
        U_name = spectral_A_name.replace('spectral_A', 'U')
        S_name = spectral_A_name.replace('spectral_A', 'S')
        V_name = spectral_A_name.replace('spectral_A', 'V')
        if spectral_B_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            spectral_A_params = lora_state_dict[spectral_A_name].to(original_params.device)
            spectral_B_params = lora_state_dict[spectral_B_name].to(original_params.device)
            spectral_C_params = lora_state_dict[spectral_C_name].to(original_params.device)
            U_params = lora_state_dict[U_name].to(original_params.device)
            S_params = lora_state_dict[S_name].to(original_params.device)
            V_params = lora_state_dict[V_name].to(original_params.device)
            lora_rank = spectral_A_params.shape[1]
            if top:
                pad_U = U_params + alpha*torch.nn.ConstantPad1d((idx*lora_rank,U_params.shape[1]-spectral_A_params.shape[1]-idx*lora_rank),0)(spectral_A_params)
                pad_S = S_params + alpha*torch.nn.ConstantPad1d((idx*lora_rank,S_params.shape[0]-spectral_C_params.shape[0]-idx*lora_rank),0)(spectral_C_params)
                pad_V = V_params + alpha*torch.nn.ConstantPad1d((idx*lora_rank,V_params.shape[1]-spectral_B_params.shape[1]-idx*lora_rank),0)(spectral_B_params)
            else:
                pad_U = U_params + alpha*torch.nn.ConstantPad1d((U_params.shape[1]-spectral_A_params.shape[1],0),0)(spectral_A_params)
                pad_S = S_params + alpha*torch.nn.ConstantPad1d((S_params.shape[0]-spectral_C_params.shape[0],0),0)(spectral_C_params)
                pad_V = V_params + alpha*torch.nn.ConstantPad1d((V_params.shape[1]-spectral_B_params.shape[1],0),0)(spectral_B_params)
            if len(original_params.shape) == 4:
                raise Exception('')
            else:
                spectral_param = pad_U@pad_S.diag()@pad_V.T
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
    updated_unet_state_dict = merge_spectralpart_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet', alpha=alpha, idx=0,top=True)
    pipe.unet.load_state_dict(updated_unet_state_dict)

    # step 3: merge lora weight to text_encoder
    text_encoder_lora_state_dict = state_dict['text_encoder']
    pretrained_text_encoder_state_dict = pipe.text_encoder.state_dict()
    updated_text_encoder_state_dict = merge_spectralpart_into_weight(pretrained_text_encoder_state_dict, text_encoder_lora_state_dict, model_type='text_encoder', alpha=alpha, idx=0,top=True)
    pipe.text_encoder.load_state_dict(updated_text_encoder_state_dict)

    return pipe, new_concept_cfg
