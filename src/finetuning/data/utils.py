"""
Utility functions for data managment and formatting.
"""

from datasets import load_dataset 
from datasets import Dataset, DatasetDict
from typing import Callable
import pandas as pd

def get_oasst_ds(
    filter: Callable | None = lambda r: r['lang']=='en',
    split: str = 'train'
) -> DatasetDict:
    """
    Load a split of the oasst dataset, potentially with a filter.
    """
    ds = load_dataset(
        'OpenAssistant/oasst1',
        split=split,
    )
    if filter is not None:
        ds = ds.filter(filter)
    return ds 

def get_conversation_roots(
    dataset: Dataset | DatasetDict | None = None    
) -> list[str]:
    """
    Get message_ids of messages with no parent, i.e. the messages that start a conversation.
    """
    if dataset is None:
        dataset = get_oasst_ds()
    return dataset.filter(
        lambda row: row['parent_id'] is None
    )['message_id']

def get_single_step_conversations(
    dataset: Dataset | DatasetDict | None = None,
    split: str = 'train' 
) -> DatasetDict:
    """
    Get conversations that consist of only the first message and all possible responses to that prompt.
    """
    if dataset is None:
        dataset = get_oasst_ds(split=split)
    roots = get_conversation_roots(dataset)
    return dataset.filter(
        lambda row: (row['message_id'] in roots) or (row['parent_id'] in roots)
    )

def create_qa_df(
    ds: DatasetDict,
    rank: int = 0
) -> pd.DataFrame:
    """
    From ds, get the questions and answer of rank rank and return a dataframe
    with the columns "prompt" and "answer", containing these strings
    """
    df = ds.filter(
        lambda row: row['parent_id'] is None or (row['rank'] is not None and row['rank'] > rank - .5 and row['rank'] < rank + .5) # 'rank' can be stored as float so we avoid direct comparison
    ).to_pandas()
    assert isinstance(df, pd.DataFrame)
    prompt_ids = df[df['parent_id'].isnull()][['message_tree_id', 'message_id', 'text']].set_index('message_tree_id')
    answer_ids = df[~df['parent_id'].isnull()][['message_tree_id', 'message_id', 'text']].set_index('message_tree_id')
    result = prompt_ids.rename(
        columns={'message_id': 'prompt_id', 'text': 'prompt'}
    ).join(
        answer_ids.rename(columns={'message_id': 'answer_id', 'text': 'answer'})
    )
    return result[['prompt_id', 'answer_id', 'prompt', 'answer']]

def create_preference_df(
    ds: DatasetDict,
    n_rows: int | None = None,
    rank_chosen: int = 0,
    rank_rejected: int = 1,
) -> Dataset:
    """
    Create a pd.DataFrame with "chosen" and "rejected" responses for a "prompt".
    This assumes single-turn conversations and can be used for DPO training. 
    """
    if n_rows is not None:
        n_rows = min([n_rows, len(set(ds['message_tree_id']))])
        ids_used = pd.Series(ds['message_tree_id']).drop_duplicates().sample(n_rows)
    else:
        ids_used = pd.Series(ds['message_tree_id']).drop_duplicates()
    ds_used = ds.filter(lambda row: row['message_tree_id'] in ids_used.values)
    chosen = create_qa_df(ds_used, rank=rank_chosen)
    rejected = create_qa_df(ds_used, rank=rank_rejected)
    res_df = chosen[['prompt', 'answer']].rename(
        columns={'answer': 'chosen'}
    ).join(
        rejected[['answer']].rename(
            columns={'answer': 'rejected'}
        ),
        how='left'
    )
    res_df = res_df[~(res_df['chosen'].isnull()) & ~(res_df['rejected'].isnull())]
    return res_df
