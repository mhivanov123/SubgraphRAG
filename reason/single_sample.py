from src.config.emb import load_yaml
import torch

from src.model.retriever import Retriever
from src.dataset.retriever import collate_retriever
from src.setup import set_seed, prepare_sample

from src.dataset.emb import customEmbInferDataset
from src.dataset.retriever import customRetrieverDataset

from preprocess.prepare_prompts import get_prompts_for_data
from prompts import icl_user_prompt, icl_ass_prompt
import os


device = torch.device(f'cuda:0')

#ENCODING MODULE

def init_text_encoder(dataset_name, config_dir):
    device = torch.device('cuda:0')
    config_file = f'{config_dir}/{dataset_name}.yaml'
    config = load_yaml(config_file)
    torch.set_num_threads(config['env']['num_threads'])

    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    return text_encoder

def embed_sample(text_encoder, sample):
    id, q_text, text_entity_list, relation_list = sample['id'], sample['question'], sample['text_entity_list'], sample['relation_list']
    q_emb, entity_embs, relation_embs = text_encoder(q_text, text_entity_list, relation_list)
    emb_dict_i = {
    'q_emb': q_emb,
    'entity_embs': entity_embs,
    'relation_embs': relation_embs
    }
    return id, emb_dict_i

def embed_question(text_encoder, question):
    q_emb = text_encoder(question)
    emb_dict_i = {
    'q_emb': q_emb
    }
    return id, emb_dict_i

#RETRIVAL MODULE

def init_retriever(retriever_path):

    cpt = torch.load(retriever_path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    emb_size = 1024 #infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def get_top_k(model, raw_sample, max_K):
    
    pred_dict = dict()

    sample = collate_retriever([raw_sample])
    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
    
    entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
    relation_list = raw_sample['relation_list']
    top_K_triples = []
    target_relevant_triples = []

    if len(h_id_tensor) != 0:
        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
        top_K_results = torch.topk(pred_triple_scores, 
                                   min(max_K, len(pred_triple_scores)))
        top_K_scores = top_K_results.values.cpu().tolist()
        top_K_triple_IDs = top_K_results.indices.cpu().tolist()

        for j, triple_id in enumerate(top_K_triple_IDs):
            top_K_triples.append((
                entity_list[h_id_tensor[triple_id].item()],
                relation_list[r_id_tensor[triple_id].item()],
                entity_list[t_id_tensor[triple_id].item()],
                top_K_scores[j]
            ))

        target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
        for triple_id in target_relevant_triple_ids:
            target_relevant_triples.append((
                entity_list[h_id_tensor[triple_id].item()],
                relation_list[r_id_tensor[triple_id].item()],
                entity_list[t_id_tensor[triple_id].item()],
            ))

    sample_dict = {
        'question': raw_sample['question'],
        'scored_triplets': top_K_triples,
        'q_entity': raw_sample['q_entity'],
        'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
        'a_entity': raw_sample['a_entity'],
        'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
        'max_path_length': raw_sample['max_path_length'],
        'target_relevant_triples': target_relevant_triples
    }

    pred_dict[raw_sample['id']] = sample_dict
    
    return sample_dict

def raw_to_pre_pred(sample, text_encoder, retrieve_model, dataset_name,k):
    
    entity_identifier_file = f"/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/{dataset_name}/entity_identifiers.txt"
    entity_identifiers = []
    with open(entity_identifier_file, 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    
    sample = customEmbInferDataset(
            sample,
            entity_identifiers).processed_dict

    id, emb_dict = embed_sample(text_encoder, sample)
    infer_set = customRetrieverDataset(sample,emb_dict)
    
    return get_top_k(retrieve_model, infer_set[0],k)

def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt

def get_outputs(outputs, model_name):
    return outputs[0]['generated_text'][-1]['content']

def llm_inf(llm, prompts, mode, model_name):
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        
        outputs = get_outputs(llm(text_inputs=conversation), model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        
        outputs = get_outputs(llm(text_inputs=conversation), model_name)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(text_inputs=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res



